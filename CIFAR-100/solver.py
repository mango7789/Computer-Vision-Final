import os
import logging
from tqdm import tqdm
from typing import Literal, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from utils import seed_everything, get_cifar_dataloader, get_cnn_model, get_vit_model, CutMix

def calculate_topk_correct(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> List[int]:
    """
    Computes the top-k correct samples for the specified values of k.

    Args:
    - output (torch.Tensor): The model predictions with shape (batch_size, num_classes).
    - target (torch.Tensor): The true labels with shape (batch_size, ).
    - topk (tuple): A tuple of integers specifying the top-k values to compute.

    Returns:
    - List of top-k correct samples for each value in topk.
    """
    maxk = max(topk)

    # get the indices of the top k predictions
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.item())
    return res


def train_with_params(
        nn_name: Literal['CNN', 'ViT'],
        epochs: int=10, 
        ft_lr: float=0.0001, 
        fc_lr: float=0.001,
        save: bool=False,
        **kwargs
    ) -> float:
    """
    Fine-Tuning the ResNet-18 and ViT(small) pretrained model on the CIFAR100 dataset with given paramters.
    
    Args:
    - nn_name: The type of the neural network, should be in ['CNN', 'ViT'].
    - epochs: Number of training epochs, default is 10. 
    - ft_lr: Fine-tuning learning rate, the learning rate except the last fc layer.
    - fc_lr: Fully-connected learning rate, the learning rate of the last fc layer.
    - save: Boolean, whether the model should be saved.
    - **kwargs: include `seed`, `data_root`, `output_dir`, `batch_size`, 
                `momentum`, `gamma`, `step_size`, `weight_decay` and `beta`.
    
    Return:
    - best_acc: The best testing accuracy during the training process.
    
    Example:
    ```python
    best_accuracy = train_with_params(nn_name='CNN', epochs=20, criterion=nn.CrossEntropyLoss())
    # NOTE: if you want to train on Kaggle, the data root should be set correctly
    best_accuracy = train_with_params(nn_name='CNN', data_root='/kaggle/input/cifar-100')
    # NOTE: if you want to disable the `CutMix`, just need to set `beta = 0`
    best_accuracy = train_with_params(nn_name='ViT', beta=0)
    ```
    """
    
    ############################################################################
    #                            initialization                                #
    ############################################################################
    
    # set the random seed to make the results reproducible
    seed = kwargs.pop('seed', 603)
    seed_everything(seed)
    
    # get data_root and output_dir
    data_root = kwargs.pop('data_root', './data/')
    output_dir = kwargs.pop('output_dir', 'logs')
    
    # get the training and testing dataloader
    batch_size = kwargs.pop('batch_size', 64)
    train_loader, test_loader = get_cifar_dataloader(root=data_root, batch_size=batch_size)
    
    # get the loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # pop other hyper-parameters from the kwargs dict
    momentum = kwargs.pop('momentum', 0.9)
    gamma = kwargs.pop('gamma', 0.5)
    step_size = kwargs.pop('step_size', 3)
    weight_decay = kwargs.pop('weight_decay', 0.0001)
    beta = kwargs.pop('beta', 1.0)
    
    # throw an error if there are extra keyword arguments
    if len(kwargs) > 0:
        extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
        raise ValueError("Unrecognized arguments %s" % extra)
    
    # get the model and corresponding optimizer
    nn_name = nn_name.lower()
    
    match nn_name:
        
        case 'cnn':
            model = get_cnn_model()
            nn_name = 'CNN'
            former_params = [p for name, p in model.named_parameters() if 'fc' not in name]
            optimizer = optim.SGD([
                    {'params': former_params, 'lr': ft_lr, 'weight_decay': weight_decay},
                    {'params': model.fc.parameters(), 'lr': fc_lr, 'weight_decay': weight_decay}
                ], momentum=momentum
            )
            
        case 'vit':
            model = get_vit_model()
            nn_name = 'ViT'
            former_params = [p for name, p in model.named_parameters() if 'classifier' not in name]
            optimizer = optim.SGD([
                    {'params': former_params, 'lr': ft_lr, 'weight_decay': weight_decay},
                    {'params': model.classifier.parameters(), 'lr': fc_lr, 'weight_decay': weight_decay}
                ], momentum=momentum
            )
            
        case _:
            raise TypeError("Please choose a valid type of neural network from ['CNN', 'ViT']!")
    
    # define scheduler for learning rate decay
    def custom_step_scheduler(optimizer: optim, epoch: int, step_size: int, gamma: float):
        """
        Decay the learning rate of the parameter group by gamma every `step_size` epochs.
        """
        if epoch % step_size == 0 and epoch > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= gamma
                
    # set the configuration for the logger
    log_directory = os.path.join(output_dir, nn_name)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    log_file_path = os.path.join(log_directory, '{}--{}--{}--{}.log'.format(epochs, ft_lr, fc_lr, batch_size))
    
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file_path),  
            logging.StreamHandler()             
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    ############################################################################
    #                                 train                                    #     
    ############################################################################
    best_acc = 0.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cutmix = CutMix(beta)
    model.to(device)
    
    # NOTE: you can change the information of logger based on your specific hyper-params
    logger.info("Train with configuration ==> ft_lr: {:>7.5f}, fc_lr: {:>7.5f}, batch_size: {:>3}".format(
        ft_lr, fc_lr, batch_size
    ))
    
    for epoch in tqdm(range(epochs)):
        '''Train'''
        model.train()
        samples = 0
        running_loss = 0.
        
        for inputs, labels in train_loader:
            
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, targets_a, targets_b, lam = cutmix(inputs, labels)
            
            if nn_name == 'ViT':
                outputs = model(pixel_values=inputs).logits
            elif nn_name == 'CNN':
                outputs = model(inputs)
            
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            
            samples += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
        
        # learning rate decay
        custom_step_scheduler(optimizer, epoch, step_size, gamma)
        
        # log the training loss
        train_loss = running_loss / samples
        logger.info("[Epoch {:>2} / {:>2}, Model {}], Training loss is {:>8.6f}".format(epoch + 1, epochs, nn_name, train_loss))

        '''Test'''
        model.eval()
        samples = 0
        running_loss = 0.
        correct_top1 = correct_top5 = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                if nn_name == 'ViT':
                    outputs = model(pixel_values=inputs).logits
                elif nn_name == 'CNN':
                    outputs = model(inputs)
                
                top1, top5 = calculate_topk_correct(outputs, labels, topk=(1, 5))
                correct_top1 += top1
                correct_top5 += top5
                samples += labels.size(0)
                
                running_loss += criterion(outputs, labels).item() * inputs.size(0)

        # add loss and accuracy to tensorboard
        test_loss = running_loss / samples
        accuracy_top1 = correct_top1 / samples
        accuracy_top5 = correct_top5 / samples
        
        logger.info("[Epoch {:>2} / {:>2}, Model {}], Testing loss is {:>8.6f}, Top-5 accuracy is {:>8.6f}, Top-1 accuracy is {:>8.6f}".format(
            epoch + 1, epochs, nn_name, test_loss, accuracy_top5, accuracy_top1
        ))
        
        # update the best accuracy and save the model if it improves
        if accuracy_top1 > best_acc:
            best_acc = accuracy_top1
            if save:
                if not os.path.exists('./model'):
                    os.mkdir('./model')
                torch.save(model.state_dict(), os.path.join('model', '{}-CIFAR100.pth'.format(nn_name)))
        
    return best_acc


def test_with_model(data_root: str='./data/', path: str='./model/CNN-CIFAR100.pth'):
    """
    Test the trained model on the CIFAR-100 dataset.
    
    Args:
    - data_root: The stored directory of the dataset.
    - path: Path to the .pth file. 
    """
    
    # get the dataset, model and loss criterion
    train_loader, test_loader = get_cifar_dataloader(root=data_root)
    nn_name = path.split('/')[2][:3]
    
    match nn_name:
        case 'CNN':
            model = get_cnn_model()
        case 'ViT':
            model = get_vit_model()
        case _:
            raise ValueError('Invalid path name!')
    
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

    
    # move the model to CUDA (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # load the trained model
    trained_state_dict = torch.load(path, map_location=device)
    model.load_state_dict(trained_state_dict)
    
    def dataset_accuracy(model, data_loader, data_type: Literal['train', 'test']):
        """
        Compute the accuracy based on the given model and dataset.
        
        Args:
        - model: The CNN or ViT model on the CIFAR-100 dataset.
        - data_loader: The train/test dataloader.
        - data_type: The type of the accuracy, should be in ['train', 'test'].
        """
        model.eval()
        class_correct_top1 = {}
        class_correct_top5 = {}
        class_samples = {}

        total_correct_top1 = 0
        total_correct_top5 = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in tqdm(data_loader):
                
                inputs, labels = inputs.to(device), labels.to(device)
                if nn_name == 'ViT':
                    outputs = model(pixel_values=inputs).logits
                elif nn_name == 'CNN':
                    outputs = model(inputs)
                
                _, pred_top1 = outputs.topk(1, 1, True, True)
                _, pred_top5 = outputs.topk(5, 1, True, True)
                pred_top1 = pred_top1.t()
                pred_top5 = pred_top5.t()
                
                correct_top1 = pred_top1.eq(labels.view(1, -1).expand_as(pred_top1))
                correct_top5 = pred_top5.eq(labels.view(1, -1).expand_as(pred_top5))
                
                total_correct_top1 += correct_top1.sum().item()
                total_correct_top5 += correct_top5.sum().item()
                total_samples += labels.size(0)
                
                # iterate each label
                for label in labels:
                    label = label.item()
                    if label not in class_correct_top1:
                        class_correct_top1[label] = 0
                        class_correct_top5[label] = 0
                        class_samples[label] = 0
                    class_correct_top1[label] += correct_top1[0, labels == label].sum().item()
                    class_correct_top5[label] += correct_top5[:, labels == label].sum().item()
                    class_samples[label] += (labels == label).sum().item()
        
        for label in sorted(class_samples.keys()):
            accuracy_top1 = class_correct_top1[label] / class_samples[label]
            accuracy_top5 = class_correct_top5[label] / class_samples[label]
            print("For class {:^30} on the CIFAR-100 dataset, Top-1 accuracy is {:>8.6f}, Top-5 accuracy is {:>8.6f}".format(
                cifar100_classes[label], accuracy_top1, accuracy_top5
            ))

        total_accuracy_top1 = total_correct_top1 / total_samples
        total_accuracy_top5 = total_correct_top5 / total_samples

        print("=" * 120)
        print("For the best model on the CIFAR-100 dataset, Total Top-1 accuracy is {:>8.6f}, Total Top-5 accuracy is {:>8.6f}".format(
            total_accuracy_top1, total_accuracy_top5
        ))

    # dataset_accuracy(model, train_loader, 'train')
    dataset_accuracy(model, test_loader, 'test')