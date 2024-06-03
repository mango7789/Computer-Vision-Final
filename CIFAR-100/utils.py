import os
import torchvision
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification


def get_cifar_dataloader(root: str='../data/', batch_size: int=64, num_workers: int=2) -> tuple[DataLoader]:
    """
    Get the train and test Dataloader of the CIFAR-100 dataset. If the dataset doesn't exist in 
    the root directory, it will be downloaded into the `root` directory automatically.
    
    Args:
    - root: The root path of the dataset, default is '../data/'
    - batch_size: The size of one min-batch of the dataloader, default is 64.
    - num_workers: How many subprocesses to use for data loading. 0 means that the data 
        will be loaded in the main process, default is 2.
    
    Return:
    - A tuple of training and testing dataloader.
    """
    
    # transform the training and testing dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
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
    model_cnn = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
    # modify the output layer
    num_features = model_cnn.fc.in_features
    model_cnn.fc = nn.Linear(num_features, 100)
    return model_cnn


def get_vit_model(path: str='WinKawaks/vit-small-patch16-224') -> ViTForImageClassification:
    """
    Return the pretrained `path` ViT model on CIFAR100 dataset.
    
    Args:
    - path: pretrained_model_name_or_path (str or os.PathLike, *optional*)
    """
    model_vit = ViTForImageClassification.from_pretrained(path, num_labels=100, ignore_mismatched_sizes=True)
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
    print(get_vit_model().classifier.parameters())