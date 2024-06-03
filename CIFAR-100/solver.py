import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Literal, List, Tuple
from utils import seed_everything, get_cifar_dataloader, get_cnn_model, get_vit_model, CutMix
# ignore the warnings
import warnings
warnings.filterwarnings('ignore')
# logger
import logging


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
        criterion: nn.Module=nn.CrossEntropyLoss(),
        save: bool=False,
        **kwargs
    ):
    """
    Fine-Tuning the ResNet-18 and ViT(small) pretrained model on the CIFAR100 dataset with given paramters.
    
    Args:
    - nn_name: The type of the neural network, should be in ['CNN', 'ViT'].
    - epochs: Number of training epochs, default is 10. 
    - ft_lr: Fine-tuning learning rate, the learning rate except the last fc layer.
    - fc_lr: Fully-connected learning rate, the learning rate of the last fc layer.
    - criterion: The loss function of the neural network.
    - save: Boolean, whether the model should be saved.
    - **kwargs: include `seed`, `batch_size`, `momentum`, `gamma`, `step_size` and other hyper-parameters.
    """
    
    ############################################################################
    #                            initialization                                #     
    ############################################################################
    
    # set the random seed to make the results reproducible
    seed = kwargs.pop('seed', 603)
    seed_everything(seed)
    
    # get the training and testing dataloader
    batch_size = kwargs.pop('batch_size', 64)
    train_loader, test_loader = get_cifar_dataloader(batch_size=batch_size)
    
    # # get the loss criterion
    # criterion = nn.CrossEntropyLoss()
    
    # pop other hyper-parameters from the kwargs dict
    momentum = kwargs.pop('momentum', 0.9)
    gamma = kwargs.pop('gamma', 0.1)
    step_size = kwargs.pop('step_size', 10)
    weight_decay = kwargs.pop('weight_decay', 1e-4)
    beta = kwargs.pop('beta', 1.0)
    
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
    log_directory = os.path.join('logs', nn_name)
    if not os.path.exists(log_directory):
        os.mkdir(log_directory)
    log_file_path = os.path.join(log_directory, '{}-{}-{}-{}-{}.log'.format(epochs, ft_lr, fc_lr, gamma, step_size))
    
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
    #                                 train                                    #     
    ############################################################################
    best_acc = 0.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cutmix = CutMix(beta)
    
    for epoch in epochs:
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
        logger.info("[Epoch {:>2} / {:>2}, Model {}], Training loss is {:>8.6f}".format(epoch + 1, nn_name, epochs, train_loss))

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
            epoch + 1, nn_name, epochs, test_loss, accuracy_top5, accuracy_top1
        ))
        
        # update the best accuracy and save the model if it improves
        if accuracy_top1 > best_acc:
            best_acc = accuracy_top1
            if save:
                if not os.path.exists('./model'):
                    os.mkdir('./model')
                torch.save(model.state_dict(), os.path.join('model', '{}-CIFAR100.pth'.format(nn_name)))