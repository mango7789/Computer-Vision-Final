import os
import logging 
from tqdm import tqdm
from typing import Tuple, Literal

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.models import resnet18

from utils import seed_everything, get_cifar_10_dataloader, get_tinyimage_dataloader, get_cifar_100_dataloader, clear_log_file
from byol import BYOL


def train_byol(
        epochs: int=100,
        lr: float=0.0003,
        hidden_dim: int=4096,
        output_dim: int=256,
        update_rate: float=0.99,
        data_type: str=Literal['cifar10', 'tinyimage'],
        save: bool=False,
        **kwargs
    ):
    """
    Train BYOL on the Tiny-ImageNet dataset with ResNet-18 as the base encoder, return
    the trained online encoder.
    
    Args:
    - epochs: The number of training epochs, default is 40.
    - lr: The learning rate of the optimizer, default is 0.0003.
    - hidden_dim: The dimension of the projection space, default is 4096.
    - output_dim: The dimension of the prediction space, default is 256.
    - update_rate: The update rate of the target by moving average, default is 0.99.
    - save: Boolean, whether the model should be saved.
    - kwargs: Contain `seed`, `data_root`, `batch_size`, `num_workers`, 
        `weight_decay` and `lr_configs`.
    """
    
    ############################################################################
    #                            initialization                                #
    ############################################################################    

    # unpack other hyper-parameters
    seed = kwargs.pop('seed', 603)
    data_root = kwargs.pop('data_root', './data/')
    output_dir = kwargs.pop('output_dir', 'logs')
    batch_size = kwargs.pop('batch_size', 64)
    num_workers = kwargs.pop('num_workers', 2)
    lr_configs = kwargs.pop('lr_configs', {})

    # throw an error if there are extra keyword arguments
    if len(kwargs) > 0:
        extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
        raise ValueError("Unrecognized arguments %s" % extra)

    # set the random seed to make the results reproducible
    seed_everything(seed)
    
    # get the dataloader
    match data_type:
        case 'cifar10': 
            train_loader = get_cifar_10_dataloader(root=data_root, batch_size=batch_size, num_workers=num_workers)
            image_size = 32
        case 'tinyimage':
            train_loader = get_tinyimage_dataloader(root=data_root, batch_size=batch_size, num_workers=num_workers)
            image_size = 64
        case _:
            raise TypeError('Please give a vaild dataset name!')
    
    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # define the model with ResNet-18 as the basic encoder
    base_encoder = resnet18(weights=None).to(device)
    base_encoder.fc = nn.Identity()
    model = BYOL(
        net=base_encoder, 
        image_size=image_size,
        hidden_layer='avgpool',
        projection_size=output_dim,
        projection_hidden_size=hidden_dim, 
        moving_average_decay=update_rate
    ).to(device)
    
    # define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, **lr_configs)
    scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.6), int(epochs * 0.8)], gamma=0.1)
    
    # set the configuration for the logger
    log_directory = os.path.join(output_dir, 'BYOL')
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    log_file_path = os.path.join(log_directory, '{}--{}--{}--{}--{}.log'.format(epochs, lr, hidden_dim, output_dim, data_type))
    
    clear_log_file(log_file_path)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file_path),  
            logging.StreamHandler()             
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    ############################################################################
    #                               training                                   #
    ############################################################################  
    
    model.train()
    for epoch in range(epochs):
        
        samples = 0
        running_loss = 0
        
        for img, _ in tqdm(train_loader):
            
            optimizer.zero_grad()
            
            img = img.to(device)
            # inspect the image from two different views  
            loss = model(img)
            
            # backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # add loss and samples
            running_loss += loss.item()
            samples += img.size(0)   
            
            # update target network
            model.update_moving_average()
        
        scheduler.step()
        training_loss = running_loss / samples
        
        logger.info("[Epoch {:>3} / {:>3}], Training loss is {:>10.8f}".format(epoch + 1, epochs, training_loss))
    
    base_encoder.fc = nn.Linear(512, 100)
    base_encoder.to(device)
    
    # define the loss function
    criterion = nn.CrossEntropyLoss()
    
    # define optimizer & scheduler
    optimizer = optim.Adam(base_encoder.parameters(), lr=0.001, **lr_configs)

    base_encoder.train()
    train_loader, _ = get_cifar_100_dataloader()
    for epoch in range(20):

        for inputs, labels in tqdm(train_loader):
            
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = base_encoder(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
                
    # save the trained byol model
    base_encoder.fc = nn.Identity()
    if save:
        save_path = 'byol.pth'
        if not os.path.exists('./model'):
            os.mkdir('./model')
        torch.save(base_encoder.state_dict(), os.path.join('./model', save_path))
    
    # close the logger 
    logging.shutdown()
        

def fine_tune_resnet18(
        epochs: int=20,
        lr: float=0.001,
        save: bool=False,
        **kwargs
    ):
    """
    Fine tune the pretrained ResNet-18 on ImageNet for CIFAR-100 classification.
    
    Args:
    - epochs: Number of training epochs, default is 20.
    - lr: Learning rate, default is 0.001.
    - save: Whether the model should be saved, default is False.
    """
    ############################################################################
    #                            initialization                                #
    ############################################################################    
    
    # unpack other hyper-parameters
    seed = kwargs.pop('seed', 603)
    data_root = kwargs.pop('data_root', './data/')
    output_dir = kwargs.pop('output_dir', 'logs')
    batch_size = kwargs.pop('batch_size', 64)
    num_workers = kwargs.pop('num_workers', 2)
    weight_decay = kwargs.pop('weight_decay', 1e-4)
    lr_configs = kwargs.pop('lr_configs', {})

    # throw an error if there are extra keyword arguments
    if len(kwargs) > 0:
        extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
        raise ValueError("Unrecognized arguments %s" % extra)
    
    # set the random seed to make the results reproducible
    seed_everything(seed)
    
    # get the dataloader
    train_loader, _ = get_cifar_100_dataloader(root=data_root, batch_size=batch_size, num_workers=num_workers)
    
    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # define the model
    model = resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 100)
    model.to(device)
    
    # define the loss function
    criterion = nn.CrossEntropyLoss()
    
    # define optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **lr_configs)
    scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.6), int(epochs * 0.8)], gamma=0.1)

    # set the configuration for the logger
    log_directory = os.path.join(output_dir, 'ResNet-18')
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    log_file_path = os.path.join(log_directory, '{}--{}--{}.log'.format(epochs, lr, batch_size))
    
    clear_log_file(log_file_path)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file_path),  
            logging.StreamHandler()             
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    ############################################################################
    #                               training                                   #
    ############################################################################  
    
    model.train()
    for epoch in range(epochs):
        
        samples = 0
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            samples += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
        
        scheduler.step()
        
        training_loss = running_loss / samples
        
        logger.info("[Epoch {:>2} / {:>2}], Training loss is {:>8.6f}".format(epoch + 1, epochs, training_loss))
    
    # save the fine-tuned resnet18 model
    if save:
        save_path = 'resnet_with_pretrain.pth'
        if not os.path.exists('./model'):
            os.mkdir('./model')
        model.fc = nn.Identity()
        torch.save(model.state_dict(), os.path.join('./model', save_path))
    
    # close the logger 
    logging.shutdown()
    

def train_resnet18(
        epochs: int=20,
        lr: float=0.001,
        save: bool=False,
        **kwargs
    ):
    """
    Train ResNet-18 from scratch on the CIFAR-100 dataset using supervised learning.
    
    Args:
    - epochs: Number of training epochs, default is 20.
    - lr: Learning rate, default is 0.001.
    - save: Whether the model should be saved, default is False.
    """
    ############################################################################
    #                            initialization                                #
    ############################################################################    

    # unpack other hyper-parameters
    seed = kwargs.pop('seed', 603)
    data_root = kwargs.pop('data_root', './data/')
    output_dir = kwargs.pop('output_dir', 'logs')
    batch_size = kwargs.pop('batch_size', 64)
    num_workers = kwargs.pop('num_workers', 2)
    weight_decay = kwargs.pop('weight_decay', 1e-4)
    lr_configs = kwargs.pop('lr_configs', {})

    # throw an error if there are extra keyword arguments
    if len(kwargs) > 0:
        extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
        raise ValueError("Unrecognized arguments %s" % extra)
    
    # set the random seed to make the results reproducible
    seed_everything(seed)
    
    # get the dataloader
    train_loader, _ = get_cifar_100_dataloader(root=data_root, batch_size=batch_size, num_workers=num_workers)
    
    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # define the model
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model.to(device)
    
    # define the loss function
    criterion = nn.CrossEntropyLoss()
    
    # define optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **lr_configs)
    scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.6), int(epochs * 0.8)], gamma=0.1)

    # set the configuration for the logger
    log_directory = os.path.join(output_dir, 'ResNet-18')
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    log_file_path = os.path.join(log_directory, '{}--{}--{}.log'.format(epochs, lr, batch_size))
    
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file_path),  
            logging.StreamHandler()             
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    ############################################################################
    #                               training                                   #
    ############################################################################  
    
    model.train()
    for epoch in range(epochs):
        
        samples = 0
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            samples += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
        
        scheduler.step()
        
        training_loss = running_loss / samples
        
        logger.info("[Epoch {:>2} / {:>2}], Training loss is {:>8.6f}".format(epoch + 1, epochs, training_loss))
    
    # save the trained resnet18 model
    if save:
        save_path = 'resnet_no_pretrain.pth'
        if not os.path.exists('./model'):
            os.mkdir('./model')
        model.fc = nn.Identity()
        torch.save(model.state_dict(), os.path.join('./model', save_path))
    
    # close the logger 
    logging.shutdown()
    

def extract_features(encoder: nn.Module, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features of the dataset after passing the `online_encoder`/`encoder`/`resnet18`(without fc layer).
    Then concatenates features and labels from different batches, enabling the downstream training or evaluation 
    processes to operate on the entire dataset seamlessly.
    
    Examples:
    ```python
    >>> features, labels = extract_features(encoder, train_loader)
    ```
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, label in tqdm(data_loader):
            images = images.to(device)
            
            feature = encoder(images).view(images.size(0), -1)
            features.append(feature.cpu())
            
            labels.append(label)
            
    return torch.cat(features, dim=0).to(device), torch.cat(labels, dim=0).to(device)


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron. Used for classifying the features extracted (learned) from the encoder.
    """
    def __init__(self, input_dim: int = 512, num_classes: int = 100):
        super(MLPClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x


def train_linear_classifier(
        train_features: torch.Tensor, train_labels: torch.Tensor, 
        test_features: torch.Tensor, test_labels: torch.Tensor,
        epochs: int, learning_rate: float, 
        type: str=Literal['self_supervise', 'supervise_with_pretrain', 'supervise_no_pretrain'],
        save: bool=False
    ) -> float:
    """
    Train a linear classifier on the training features and labels, then test the classifier on 
    the testing features and labels.
    """
    ############################################################################
    #                            initialization                                #
    ############################################################################ 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize the linear classifier
    input_dim = train_features.shape[1]
    classifier = MLPClassifier(input_dim).to(device)

    # define criterion, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=0.0008)


    # set the configuration for the logger
    log_directory = os.path.join('./logs', 'Linear-Classifier')
    os.makedirs(log_directory, exist_ok=True)
    log_file_path = os.path.join(log_directory, '{}.log'.format(type))
    
    clear_log_file(log_file_path)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file_path),  
            logging.StreamHandler()             
        ]
    )
    
    logger = logging.getLogger(__name__)

    ############################################################################
    #                               train                                      #
    ############################################################################ 
    best_accuracy = 0.
    
    # convert labels to long tensor (required by CrossEntropyLoss)
    train_labels = train_labels.long()
    test_labels = test_labels.long()
    
    
    for epoch in range(epochs):
        
        # train the classifier
        classifier.train()
        optimizer.zero_grad()
        
        outputs = classifier(train_features)
        
        loss = criterion(outputs, train_labels)
        loss.backward()
        
        optimizer.step()
        
        logger.info("[Epoch {:>3} / {:>3}], Training loss is {:>8.6f}".format(epoch + 1, epochs, loss.item()))

        # evaluate the classifier
        classifier.eval()
        with torch.no_grad():
            
            outputs = classifier(test_features)
            _, predicted = torch.max(outputs, 1)
            
            # compute validation loss
            criterion_ = nn.CrossEntropyLoss()
            validation_loss = criterion_(outputs, test_labels).item()
            
            # compute accuracy
            accuracy = (predicted == test_labels).float().mean().item()
            
            logger.info("[Epoch {:>3} / {:>3}], Validation loss is {:>8.6f}, Validation accuracy is {:>8.6f}".format(
                epoch + 1, epochs, validation_loss, accuracy 
            ))
            
            # update the best accuracy and save the model if it improves
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                if save:
                    if not os.path.exists('./model'):
                        os.mkdir('./model')
                    torch.save(classifier.state_dict(), os.path.join('model', '{}.pth'.format(type)))
    
    # close the logger 
    logging.shutdown()        
            
    return best_accuracy


def test_trained_model(encoder_path: str, classifier_path: str):
    """
    Load and test the trained encoder and linear classifier on the testing
    dataset of CIFAR-100.
    
    Args:
    - encoder_path: The file path of the learned parameters of encoder model.
    - classifier_path: The file path of the learned parameters of linear classifier.
    """
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load the encoder
    encoder = resnet18(weights=None)
    encoder.fc = nn.Identity()
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device)
    encoder.eval()
    
    # load the linear classifier
    classifier = MLPClassifier(512)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    
    # labels of CIFAR-100
    cifar100_classes = [
        "beaver", "dolphin", "otter", "seal", "whale",
        "aquarium fish", "flatfish", "ray", "shark", "trout",
        "orchids", "poppies", "roses", "sunflowers", "tulips",
        "bottles", "bowls", "cans", "cups", "plates",
        "apples", "mushrooms", "oranges", "pears", "sweet peppers",
        "clock", "computer keyboard", "lamp", "telephone", "television",
        "bed", "chair", "couch", "table", "wardrobe",
        "bee", "beetle", "butterfly", "caterpillar", "cockroach",
        "bear", "leopard", "lion", "tiger", "wolf",
        "bridge", "castle", "house", "road", "skyscraper",
        "cloud", "forest", "mountain", "plain", "sea",
        "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
        "fox", "porcupine", "possum", "raccoon", "skunk",
        "crab", "lobster", "snail", "spider", "worm",
        "baby", "boy", "girl", "man", "woman",
        "crocodile", "dinosaur", "lizard", "snake", "turtle",
        "hamster", "mouse", "rabbit", "shrew", "squirrel",
        "maple", "oak", "palm", "pine", "willow",
        "bicycle", "bus", "motorcycle", "pickup truck", "train",
        "lawn-mower", "rocket", "streetcar", "tank", "tractor"
    ]
    
    # test on the testing dataset
    _, test_loader = get_cifar_100_dataloader()
    
    correct = 0
    total = 0
    correct_per_class = [0] * 100
    total_per_class = [0] * 100
    
    # test the model
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            features = encoder(images)
            outputs = classifier(features)
            
            # calculate predictions
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for label, prediction in zip(labels, predicted):
                total_per_class[label] += 1
                if prediction == label:
                    correct_per_class[label] += 1
                    
    file_path = f"./logs/{encoder_path.split('/')[-1].split('.')[0]}-accuracy.txt"

    # clear the content in the file if it exists
    clear_log_file(file_path)
    
    # print the result on console and save them in .txt file
    with open(file_path, "w") as file:
        # each label
        for i in range(len(cifar100_classes)):
            accuracy = correct_per_class[i] / total_per_class[i]
            file.write("For class {:^30} on the CIFAR-100 dataset, Testing accuracy is {:>8.6f}\n".format(
                cifar100_classes[i], accuracy
            ))
            print("For class {:^30} on the CIFAR-100 dataset, Testing accuracy is {:>8.6f}".format(
                cifar100_classes[i], accuracy
            ))
            
        # total accuracy
        total_accuracy = correct / total

        file.write("=" * 121 + "\n")
        file.write("For the best model on the CIFAR-100 dataset, Total testing accuracy is {:>8.6f}\n".format(
            total_accuracy
        ))
        print("=" * 121 + "\n")
        print("For the best model on the CIFAR-100 dataset, Total testing accuracy is {:>8.6f}".format(
            total_accuracy
        ))
    
    return