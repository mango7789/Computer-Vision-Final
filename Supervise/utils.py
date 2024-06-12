import os
import random
import zipfile
import urllib.request
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

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


class GaussianBlur(object):
    """
    Blur a single image on CPU. Modified from the source code in 
    https://github.com/sthalles/SimCLR/blob/master/data_aug/gaussian_blur.py.
    """
    def __init__(self, kernel_size: float | int):
        
        # make the kernel size an odd and positive integer 
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        
        # horizontal
        self.blur_h = nn.Conv2d(
            3, 3, kernel_size=(kernel_size, 1),
            stride=1, padding=0, bias=False, groups=3
        )
        
        # vertical
        self.blur_v = nn.Conv2d(
            3, 3, kernel_size=(1, kernel_size),
            stride=1, padding=0, bias=False, groups=3
        )
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        """
        Convert PIL image to tensor, apply GaussianBlur on it. Then reconvert it to PIL image.
        """
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img
    
    
class ContrastiveLearningViewGenerator(object):
    """
    Take two random crops of one image as the query and key. Modified from the source code
    in https://github.com/sthalles/SimCLR/blob/master/data_aug/view_generator.py.
    """
    def __init__(self, base_transform, n_views: int=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]
    
    
def get_cifar_10_dataloader(root: str='./data', batch_size: int=64, num_workers: int=2) -> DataLoader:
    """
    Get the train dataloader of the CIFAR-10 dataset.
    """
    
    # transform the training image to tensor
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=int(0.1 * 32)),
        transforms.ToTensor()
    ])
    
    # if the dataset doesn't exist in local, download it automatically
    download = (not os.path.exists(root)) or ('cifar-10-batches-py' not in os.listdir(root))
    
    # get the training and testing dataloader
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=ContrastiveLearningViewGenerator(transform))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    
    return train_loader

    
class TinyImageNetDataset(Dataset):
    """
    A subclass of class `Dataset` implemented in `torch.utils.data`, used specifically for the
    Tiny-ImageNet-200 dataset.
    """
    def __init__(self, root_dir: str, transform: transforms.Compose):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self._download_extract()
        self._load_train()

    def _download_extract(self):
        """
        Download the dataset in local if it doesn't exist and unzip the dataset.
        """
        data_dir = os.path.join(self.root_dir, 'tiny-imagenet-200')
        zip_file = os.path.join(self.root_dir, 'tiny-imagenet-200.zip')
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

        def download_url(url :str, output_path :str):
            """
            Download the Tiny-ImageNet dataset from the given url.
            """
            response = urllib.request.urlopen(url)
            total = int(response.getheader('Content-Length').strip())
            if not os.path.exists(self.root_dir):
                os.mkdir(self.root_dir)
            with open(output_path, 'wb') as f, tqdm(
                desc='Downloading',
                total=total,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response:
                    size = f.write(data)
                    bar.update(size)

        def extract_zip(file_path :str, extract_to :str):
            """
            Extract downloaded zip files into the `extract_to` directory.
            """
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                for member in tqdm(
                    iterable=zip_ref.infolist(),
                    total=len(zip_ref.infolist()),
                    desc='Extracting ',
                    unit='file(s)'
                ):
                    zip_ref.extract(member, extract_to)
                    
        if not os.path.exists(data_dir):
            # download Tiny ImageNet
            if not os.path.exists(zip_file):
                print("Downloading {} to {}".format(url, zip_file))
                download_url(url, zip_file)
            # extract the dataset
            print("Extracting {} to {}".format(zip_file, data_dir))
            extract_zip(zip_file, 'data')
        
        # delete the zip file
        if os.path.exists(zip_file):
            os.remove(zip_file)
            print("Deleted zip file {}".format(zip_file))
        
    def _load_train(self):
        """
        Load the training dataset of the Tiny-ImageNet. We only need training dataset here.
        """
        train_dir = os.path.join(self.root_dir, 'tiny-imagenet-200', 'train')
        classes = os.listdir(train_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        # iterate each class, append image path and corresponding label
        for cls in classes:
            cls_dir = os.path.join(train_dir, cls, 'images')
            images = os.listdir(cls_dir)
            for img_name in images:
                img_path = os.path.join(cls_dir, img_name)
                self.data.append(img_path)
                self.labels.append(self.class_to_idx[cls])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[int]]:
        img_path = self.data[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        # transform the original image
        image = self.transform(image)
        
        return image, label 
        
    
def get_tinyimage_dataloader(root: str='./data', batch_size: int=64, num_workers: int=2) -> DataLoader:
    """
    Get the train Dataloader of the Tiny-ImageNet dataset. If the dataset doesn't exist in local, it will
    be downloaded and extracted into the `root` directory automatically.
    
    Args:
    - root: The root path of the dataset, default is './data/'
    - batch_size: The size of one min-batch of the dataloader, default is 64.
    - num_workers: How many subprocesses to use for data loading. 0 means that the data 
        will be loaded in the main process, default is 2.
    
    Return:
    - train_loader: The Dataloader of the training dataset.
    """
    
    # define data augmentation and normalization for training dataset
    transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=int(0.1 * 64)),
        transforms.ToTensor()
    ])
    
    # get the training dataloader
    train_set = TinyImageNetDataset(root_dir=root, transform=ContrastiveLearningViewGenerator(transform))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    
    return train_loader


def get_cifar_100_dataloader(root: str='./data', batch_size: int=64, num_workers: int=2) -> Tuple[DataLoader]:
    """
    Inherited from the `utils.get_cifar_dataloader` function from Task2.
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
    download = (not os.path.exists(root)) or ('cifar-100-python' not in os.listdir(root))
    
    # get the training and testing dataloader
    train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader


def clear_log_file(log_file_path: str):
    """
    Check and clear the log file if it exists and is not empty
    """
    if os.path.exists(log_file_path):
        if os.path.getsize(log_file_path) > 0:
            os.remove(log_file_path)


if __name__ == '__main__':
    # if you just want to download dataset on local, run this script
    get_cifar_10_dataloader()
    get_tinyimage_dataloader()
    get_cifar_100_dataloader()