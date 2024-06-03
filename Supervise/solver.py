import os
import logging 
from tqdm import tqdm
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from byol import BYOL
from utils import seed_everything, get_places365_dataloader


def train_byol(
        epochs: int=10,
        lr :float=0.003,
        proj_dim :int=1000,
        pred_dim :int=1000,
        save :bool=False,
        **kwargs
    ) -> BYOL:
    """
    Train BYOL on the Places-365 dataset with ResNet-18 as the base encoder.
    
    Args:
    - epochs: The number of training epochs, default is 10.
    - lr: The learning rate of the optimizer, default is 0.003.
    - proj_dim: The dimension of the projection space, default is 1000.
    - pred_dim: The dimension of the prediction space, default is 1000.
    - save: Boolean, whether the model should be saved.
    - kwargs: Contain `seed`, `data_root`, `batch_size`, `num_workers`, 
            `lr_configs` and `update_rate`.
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
    update_rate = kwargs.pop('update_rate', 0.996)

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
    base_encoder = models.resnet18(pretrained=False)
    base_encoder.fc = nn.Identity()
    model = BYOL(base_encoder, projection_dim=proj_dim, prediction_dim=pred_dim).to(device)
    
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, **lr_configs)
    
    # set the configuration for the logger
    log_directory = os.path.join(output_dir, 'byol')
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    log_file_path = os.path.join(log_directory, '{}-{}-{}-{}.log'.format(epochs, lr, proj_dim, pred_dim))
    
    logging.basicConfig(
        level=logging.DEBUG,
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
    for epoch in tqdm(epochs):
        total_loss = 0
        for images, _ in train_loader:
            # inspect the image from two different views
            img1, img2 = images[0].to(device), images[1].to(device)

            optimizer.zero_grad()
            
            pred1, pred2, target1, target2 = model(img1, img2)
            loss = BYOL.loss(pred1, pred2, target1, target2)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        training_loss = total_loss / len(train_loader)
        
        logger.info("[Epoch {:>2} / {:>2}], Training loss is {:>8.6f}".format(epoch + 1, epochs, training_loss))
        
        # update target network
        with torch.no_grad():
            for param_online, param_target in zip(model.online_encoder.parameters(), model.target_encoder.parameters()):
                param_target.data = param_target.data * update_rate + param_online.data * (1 - update_rate)
    
    # save the trained byol model
    if save:
        save_path = 'byol.pth'
        if not os.path.exists('./model'):
            os.mkdir('./model')
            torch.save(model.state_dict(), os.path.join('./model', save_path))
    
    return model


def extract_features(model: nn.Module, data_loader: DataLoader) -> Tuple[torch.Tensor]:
    """
    Extract features of the dataset after passing the `online_encoder`.
    """
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, label in data_loader:
            images = images.cuda()
            
            feature = model.online_encoder[0](images).view(images.size(0), -1)
            features.append(feature.cpu())
            
            labels.append(label)
            
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def train_linear_classifier(
        train_features :torch.Tensor, 
        train_labels :torch.Tensor, 
        test_features :torch.Tensor,
        test_labels :torch.Tensor
    ) -> float:
    """
    Train a linear classifier on the training features and labels, then test the classifier on 
    the testing features and labels. Return the accuracy of the classifier. The linear classifier 
    is choosen as `LogisticRegression`.
    """
    # fit on the training set
    linear_classifier = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
    linear_classifier.fit(train_features, train_labels)

    # evaluate on the testing set
    test_preds = linear_classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, test_preds)
    
    return accuracy