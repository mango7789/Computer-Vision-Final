import os
import logging 
from tqdm import tqdm
from typing import Tuple, Literal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from model import Encoder, BYOL
from utils import seed_everything, get_tinyimage_dataloader, get_cifar_100_dataloader, self_supervise_augumentation


def train_byol(
        epochs: int=30,
        lr :float=0.001,
        hidden_dim :int=4096,
        output_dim :int=256,
        update_rate: float=0.996,
        save :bool=False,
        **kwargs
    ) -> Encoder:
    """
    Train BYOL on the Tiny-ImageNet dataset with ResNet-18 as the base encoder, return
    the trained online encoder.
    
    Args:
    - epochs: The number of training epochs, default is 10.
    - lr: The learning rate of the optimizer, default is 0.003.
    - hidden_dim: The dimension of the projection space, default is 4096.
    - output_dim: The dimension of the prediction space, default is 256.
    - update_rate: The update rate of the target by moving average.
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
    weight_decay = kwargs.pop('weight_decay', 1e-4)
    lr_configs = kwargs.pop('lr_configs', {})

    # throw an error if there are extra keyword arguments
    if len(kwargs) > 0:
        extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
        raise ValueError("Unrecognized arguments %s" % extra)

    # set the random seed to make the results reproducible
    seed_everything(seed)
    
    # get the dataloader
    train_loader = get_tinyimage_dataloader(root=data_root, batch_size=batch_size, num_workers=num_workers)
    
    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # define the model with ResNet-18 as the basic encoder
    model = BYOL(Encoder, hidden_dim, output_dim, momentum=update_rate).to(device)
    
    # define optimizer
    # NOTE: since we use adam as the optimizer, the learning rate will be adjusted dynamically, so 
    #       no additional scheduler is applied here
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **lr_configs)

    # set the configuration for the logger
    log_directory = os.path.join(output_dir, 'BYOL')
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    log_file_path = os.path.join(log_directory, '{}--{}--{}--{}.log'.format(epochs, lr, hidden_dim, output_dim))
    
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
    
    # get data augmentation
    augmentation = self_supervise_augumentation()
    
    ############################################################################
    #                               training                                   #
    ############################################################################  
    
    model.train()
    for epoch in tqdm(range(epochs)):
        # TODO: The calculation of the loss.
        samples = 0
        running_loss = 0
        for images, _ in train_loader:
            # inspect the image from two different views        
            for img in images:
                img1, img2 = augmentation(img).to(device), augmentation(img).to(device)    

                optimizer.zero_grad()
                
                pred1, pred2, target1, target2 = model(img1, img2)
                loss = BYOL.loss(pred1, pred2, target1, target2)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

            samples += images.size(0)   # add the batch size
        
        training_loss = running_loss / samples
        
        logger.info("[Epoch {:>2} / {:>2}], Training loss is {:>8.6f}".format(epoch + 1, epochs, training_loss))
        
        # update target network
        model.update_target_network()
    
    # save the trained byol model
    if save:
        save_path = 'byol.pth'
        if not os.path.exists('./model'):
            os.mkdir('./model')
            torch.save(model.state_dict(), os.path.join('./model', save_path))
    
    return model.online_encoder


def fetch_resnet18() -> Encoder:
    """
    Return the pretrained ResNet-18 on ImageNet.
    """
    return Encoder(pretrain=True)


def train_resnet18(
        epochs: int=30,
        lr :float=0.001,
        save: bool=False,
        **kwargs
    ):
    """
    Train ResNet-18 from scratch on the CIFAR-100 dataset using supervised learning.
    
    Args:
    - epochs: Number of training epochs, default is 30.
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
    train_loader = get_cifar_100_dataloader(root=data_root, batch_size=batch_size, num_workers=num_workers)
    
    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # define the model
    model = Encoder().to(device)
    
    # define the loss function
    criterion = nn.CrossEntropyLoss()
    
    # define optimizer & scheduler
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, **lr_configs)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

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
    for epoch in tqdm(range(epochs)):
        
        samples = 0
        running_loss = 0.0
        for inputs, labels in train_loader:
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
        save_path = 'resnet18.pth'
        if not os.path.exists('./model'):
            os.mkdir('./model')
            torch.save(model.state_dict(), os.path.join('./model', save_path))
    
    return model


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
        for images, label in data_loader:
            images = images.to(device)
            
            feature = encoder(images).view(images.size(0), -1)
            features.append(feature.cpu())
            
            labels.append(label)
            
    return torch.cat(features, dim=0).to(device), torch.cat(labels, dim=0).to(device)


class LinearClassifier(nn.Module):
    """
    Use a traditional MLP as the lienar classifier protocol.
    """
    def __init__(self, input_dim :int=512, num_classes :int=100):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x :torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_linear_classifier(
        train_features :torch.Tensor, train_labels :torch.Tensor, 
        test_features :torch.Tensor, test_labels :torch.Tensor,
        epochs: int=100, learning_rate: float=0.001, 
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
    classifier = LinearClassifier(input_dim).to(device)

    # define criterion, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)

    # set the configuration for the logger
    log_directory = os.path.join('./log', 'linear_classifier')
    os.makedirs(log_directory, exist_ok=True)
    log_file_path = os.path.join(log_directory, '{}.log'.format(type))
    
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
        scheduler.step()
        
        logger.info("[Epoch {:>2} / {:>2}], Training loss is {:>8.6f}".format(epoch + 1, epochs, loss / train_features.size(0)))

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
            
            logger.info("[Epoch {:>2} / {:>2}], Validation loss is {:>8.6f}, Validation accuracy is {:>8.6f}".format(
                epoch + 1, epochs, accuracy, validation_loss 
            ))
            
            # update the best accuracy and save the model if it improves
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                if save:
                    if not os.path.exists('./model'):
                        os.mkdir('./model')
                    torch.save(classifier.state_dict(), os.path.join('model', '{}.pth'.format(type)))
            
    return best_accuracy