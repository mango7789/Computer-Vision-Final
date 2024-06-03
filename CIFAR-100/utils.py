import os
import random
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTConfig


def seed_everything(seed: int=None):
    """
    Set the random seed for the whole neural network.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_cifar_dataloader(root: str='./data/', batch_size: int=64, num_workers: int=2) -> tuple[DataLoader]:
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
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), 
            std=(0.2675, 0.2565, 0.2761)
        )    
    ])

    transform_test = transforms.Compose([
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
    Return the pretrained Resnet-50 model on CIFAR100 dataset.
    """
    class ModifiedResNet50(nn.Module):
        def __init__(self, num_classes :int=1000): 
            super(ModifiedResNet50, self).__init__()
            self.resnet50 = models.resnet50(pretrained=True)
            
            # modify the first convolutional layer to accept 32x32 input images
            self.resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet50.bn1 = nn.BatchNorm2d(64)
            self.resnet50.relu = nn.ReLU(inplace=True)
            self.resnet50.maxpool = nn.Identity()  # Remove the maxpool layer to avoid reducing the spatial dimensions too much

            # adjust the number of input features for the fully connected layer
            num_ftrs = self.resnet50.fc.in_features
            self.resnet50.fc = nn.Linear(num_ftrs, num_classes)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.resnet50.conv1(x)
            x = self.resnet50.bn1(x)
            x = self.resnet50.relu(x)
            # skip the maxpool layer since it's replaced by nn.Identity()
            x = self.resnet50.layer1(x)
            x = self.resnet50.layer2(x)
            x = self.resnet50.layer3(x)
            x = self.resnet50.layer4(x)
            x = self.resnet50.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.resnet50.fc(x)
            return x
        
    return ModifiedResNet50().resnet50

def get_vit_model(path: str='WinKawaks/vit-small-patch16-224') -> ViTForImageClassification:
    """
    Return the pretrained `path` ViT model on CIFAR100 dataset.
    
    Args:
    - path: pretrained_model_name_or_path (str or os.PathLike, *optional*)
    """
    config = ViTConfig.from_pretrained(path)
    
    # adjust the configuration for 32x32 input images and number of clases in CIFAR100
    config.image_size = 32   
    config.hidden_size = 384 
    config.num_channels = 3   
    config.num_classes = 100  
    
    # load the modified model
    model_vit = ViTForImageClassification(config)  
      
    return model_vit


def count_model_parameters(model: nn.Module) -> int:
    """
    Return the number of parameters in the given `model`, the model should be instantiated.
    
    Args:
    - model: The instantiated model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':    
    # print(count_model_parameters(get_cnn_model()))
    # print(count_model_parameters(get_vit_model()))
    print(get_vit_model())
