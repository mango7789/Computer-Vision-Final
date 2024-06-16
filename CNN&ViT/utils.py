import os
import random
from typing import List, Tuple

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models
from transformers import ViTForImageClassification


def seed_everything(seed: int=None):
    """
    Set the random seed for the whole neural network.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_cifar_dataloader(root: str='./data/', batch_size: int=64, num_workers: int=2) -> Tuple[DataLoader]:
    """
    Get the train and test Dataloader of the CIFAR-100 dataset. If the dataset doesn't exist in 
    the root directory, it will be downloaded into the `root` directory automatically.
    
    Args:
    - root: The root path of the dataset, default is './data/'
    - batch_size: The size of one min-batch of the dataloader, default is 64.
    - num_workers: How many subprocesses to use for data loading. 0 means that the data 
        will be loaded in the main process, default is 2.
    
    Return:
    - A tuple of training and testing dataloader.
    """
    
    # transform the training and testing dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), 
            std=(0.2675, 0.2565, 0.2761)
        )    
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), 
            std=(0.2675, 0.2565, 0.2761)
        )
    ])
    
    # if the dataset doesn't exist in local, download it automatically
    download = (not os.path.exists(root)) or (os.listdir(root) == [])
    
    # get the training and testing dataloader
    train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader


def get_cnn_model() -> models.resnet50:
    """
    Return the pretrained Resnet-50 model on CIFAR-100 dataset.
    """
    model_cnn = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
    # modify the output layer
    num_features = model_cnn.fc.in_features
    model_cnn.fc = nn.Linear(num_features, 100)
    return model_cnn


def get_vit_model(path: str='WinKawaks/vit-small-patch16-224') -> ViTForImageClassification:
    """
    Return the pretrained `path` ViT model on CIFAR-100 dataset.
    
    Args:
    - path: pretrained_model_name_or_path (str or os.PathLike, *optional*)
    """
    model_vit = ViTForImageClassification.from_pretrained(path, num_labels=100, ignore_mismatched_sizes=True)
    return model_vit


class CutMix:
    """
    CutMix for data augumentation of the CIFAR-100 dataset.
    """
    def __init__(self, beta: float=1.0):
        self.beta = beta

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        
        if self.beta > 0 and torch.rand(1).item() > 0.5:
            # sample lambda from a beta distribution
            lam = torch.distributions.beta.Beta(self.beta, self.beta).sample().item()
            
            # randomly shuffle the batch
            rand_index = torch.randperm(x.size()[0])
            target_a = y
            target_b = y[rand_index]
            
            # replace a portion of the input image with another image
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            
            # adjust lambda to consider the clipped area
            lam = 1 - ((bbx2 - bbx1) * (bbx2 - bbx1) / (x.size()[-1] * x.size()[-2]))
            return x, target_a, target_b, lam
        
        return x, y, y, 1.0

    def _rand_bbox(self, size: torch.Tensor, lam: float) -> Tuple:
        """
        Generate a random bounding box for the images to apply CutMix, return the 
        (x, y) coordinates of the bounding points.
        """
        W = size[2]
        H = size[3]
        cut_rat = torch.sqrt(torch.tensor(1.0 - lam))
        cut_w = torch.tensor(int(W * cut_rat)).item()
        cut_h = torch.tensor(int(H * cut_rat)).item()

        # generate random center of point
        cx = torch.randint(0, W, (1,)).item()
        cy = torch.randint(0, H, (1,)).item()

        # calculate the coordinates of the bbox
        bbx1 = torch.clamp(torch.tensor(cx) - cut_w // 2, 0, W)
        bby1 = torch.clamp(torch.tensor(cy) - cut_h // 2, 0, H)
        bbx2 = torch.clamp(torch.tensor(cx) + cut_w // 2, 0, W)
        bby2 = torch.clamp(torch.tensor(cy) + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


def clear_log_file(log_file_path: str):
    """
    Check and clear the log file if it exists and is not empty
    """
    if os.path.exists(log_file_path):
        if os.path.getsize(log_file_path) > 0:
            os.remove(log_file_path)


def count_model_parameters(model: nn.Module) -> int:
    """
    Return the number of parameters in the given `model`, the model should be instantiated.
    
    Args:
    - model: The instantiated model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    cnn = get_cnn_model()
    vit = get_vit_model()
    print(count_model_parameters(cnn))
    print(count_model_parameters(vit))