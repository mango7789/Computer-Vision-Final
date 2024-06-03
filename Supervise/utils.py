import os
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import List, Tuple

def seed_everything(seed: int=None):
    """
    Set the random seed for the whole neural network.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

def get_places365_dataloader(root: str='./data/', batch_size: int=64, num_workers: int=2) -> Tuple[DataLoader]:
    """
    Get the train and test Dataloader of the Places-365 dataset. If the dataset doesn't exist in 
    the root directory, it will be downloaded into the `root` directory automatically.
    
    Args:
    - root: The root path of the dataset, default is './data/'
    - batch_size: The size of one min-batch of the dataloader, default is 64.
    - num_workers: How many subprocesses to use for data loading. 0 means that the data 
        will be loaded in the main process, default is 2.
    
    Return:
    - A tuple of training and testing dataloader.
    """
    
    # define data augmentation and normalization for training and validation dataset
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # if the dataset doesn't exist in local, download it automatically
    download = (not os.path.exists(root)) or (os.listdir(root) == [])
    
    # get the training and testing dataloader
    train_set = torchvision.datasets.Places365(root=root, split='train-standard', download=download, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = torchvision.datasets.Places365(root=root, split='val', download=download, transform=val_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader