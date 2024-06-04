import os
import logging 
from tqdm import tqdm
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from byol import Encoder, BYOL
from utils import seed_everything, get_places365_dataloader, get_cifar_dataloader, self_supervise_augumentation


def train_byol(
        epochs: int=10,
        lr :float=0.003,
        hidden_dim :int=4096,
        output_dim :int=256,
        update_rate: float=0.996,
        save :bool=False,
        **kwargs
    ) -> Encoder:
    """
    Train BYOL on the Places-365 dataset with ResNet-18 as the base encoder, return
    the trained online encoder.
    
    Args:
    - epochs: The number of training epochs, default is 10.
    - lr: The learning rate of the optimizer, default is 0.003.
    - hidden_dim: The dimension of the projection space, default is 4096.
    - output_dim: The dimension of the prediction space, default is 256.
    - update_rate: The update rate of the target by moving average.
    - save: Boolean, whether the model should be saved.
    - kwargs: Contain `seed`, `data_root`, `batch_size`, `num_workers`, 
            and `lr_configs`.
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
    train_loader = get_places365_dataloader(root=data_root, batch_size=batch_size, num_workers=num_workers)
    
    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # define the model with ResNet-18 as the basic encoder
    model = BYOL(Encoder(), hidden_dim, output_dim, momentum=update_rate).to(device)
    
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, **lr_configs)

    # set the configuration for the logger
    log_directory = os.path.join(output_dir, 'byol')
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    log_file_path = os.path.join(log_directory, '{}-{}-{}-{}.log'.format(epochs, lr, hidden_dim, output_dim))
    
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
            # TODO: should we move `batch` or not?
            # inspect the image from two different views            
            img1 = torch.stack([augmentation(image) for image in images], dim=0).to(device)
            img2 = torch.stack([augmentation(image) for image in images], dim=0).to(device)

            optimizer.zero_grad()
            
            pred1, pred2, target1, target2 = model(img1, img2)
            loss = BYOL.loss(pred1, pred2, target1, target2)
            
            loss.backward()
            optimizer.step()

            samples += images.size(0)
            running_loss += loss.item()
        
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
        epochs: int=10,
        lr :float=0.001,
        save: bool=False,
        **kwargs
    ):
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
    train_loader = get_cifar_dataloader(root=data_root, batch_size=batch_size, num_workers=num_workers)
    
    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # define the model
    model = Encoder().to(device)
    
    # define the loss function
    criterion = nn.CrossEntropyLoss()
    
    # define optimizer & scheduler
    optimizer = optim.SGD(model.parameters(), lr=lr, **lr_configs)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    # set the configuration for the logger
    log_directory = os.path.join(output_dir, 'resnet18')
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    log_file_path = os.path.join(log_directory, '{}-{}-{}.log'.format(epochs, lr, batch_size))
    
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
        
        # update target network
        model.update_target_network()
    
    # save the trained resnet18 model
    if save:
        save_path = 'resnet18.pth'
        if not os.path.exists('./model'):
            os.mkdir('./model')
            torch.save(model.state_dict(), os.path.join('./model', save_path))
    
    return model


def extract_features(encoder: nn.Module, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features of the dataset after passing the `online_encoder`.
    
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
            
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def train_linear_classifier(
        train_features :torch.Tensor, train_labels :torch.Tensor, 
        test_features :torch.Tensor, test_labels :torch.Tensor
    ) -> float:
    """
    Train a linear classifier on the training features and labels, then test the classifier on 
    the testing features and labels. Return the accuracy of the classifier. The linear classifier 
    is choosen as `LogisticRegression`.
    """
    # fit on the training set
    linear_classifier = LogisticRegression(max_iter=500, solver='lbfgs', multi_class='multinomial')
    linear_classifier.fit(train_features, train_labels)

    # evaluate on the testing set
    test_preds = linear_classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, test_preds)
    
    return accuracy