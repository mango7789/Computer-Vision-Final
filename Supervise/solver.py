import os
import torch
import torch.nn as nn
import torchvision.models as models
from byol import BYOL
from utils import seed_everything, get_places365_dataloader

import logging 
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def train_byol(
        epochs: int=10,
        lr :float=0.003,
        proj_dim :int=512,
        pred_dim :int=2048,
        **kwargs
    ):
    """
    Train BYOL on the Places-365 dataset with ResNet-18 as the base encoder.
    
    Args:
    - epochs: The number of training epochs, default is 10.
    - lr: The learning rate of the optimizer, default is 0.003.
    - proj_dim: The dimension of the projection space, default is 512.
    - pred_dim: The dimension of the prediction space, default is 2048.
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
    train_loader, _ = get_places365_dataloader(root=data_root, batch_size=batch_size, num_workers=num_workers)
    
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
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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
            img1, img2 = images[0].to(device), images[1].to(device)

            optimizer.zero_grad()
            
            pred1, pred2, target1, target2 = model(img1, img2)
            loss = BYOL.loss(pred1, pred2, target1, target2)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        logger.info("[Epoch {:>2} / {:>2}], Training loss is {:>8.6f}".format(epoch + 1, epochs, avg_loss))
        
        # update target network
        with torch.no_grad():
            for param_online, param_target in zip(model.online_encoder.parameters(), model.target_encoder.parameters()):
                param_target.data = param_target.data * update_rate + param_online.data * (1 - update_rate)