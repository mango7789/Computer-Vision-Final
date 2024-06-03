import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils import get_cifar_dataloader, get_cnn_model, get_vit_model

# ignore the warnings
import warnings
warnings.filterwarnings('ignore')



def train_with_params(
        epoch: int=10, 
        ft_lr: float=0.0001, 
        fc_lr: float=0.001,
        **kwargs
    ):
    """
    Fine-Tuning the ResNet-18 and ViT(small) pretrained model on the CIFAR100 dataset with given paramters.
    
    Args:
    - epoch: Number of training epochs, default is 10. 
    - ft_lr: Fine-tuning learning rate, the learning rate except the last fc layer.
    - fc_lr: Fully-connected learning rate, the learning rate of the last fc layer.
    - **kwargs: include `batch_size`, `momentum`, `gamma`, `step_size` and other hyper-parameters.
    """
    
    # get the training and testing dataloader
    batch_size = kwargs.pop('batch_size', 64)
    train_loader, test_loader = get_cifar_dataloader(batch_size=batch_size)
    
    # get the loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # get the model
    cnn_model, vit_model = get_cnn_model(), get_vit_model()
    
    # pop other hyper-parameters from the kwargs dict
    momentum = kwargs.pop('momentum', 0.9)
    gamma = kwargs.pop('gamma', 0.1)
    step_size = kwargs.pop('step_size', 10)
    weight_decay = kwargs.pop('weight_decay', 1e-4)
    # optim_rule = kwargs.pop('optim_rule', 'sgd')
    
    # get params except the last fc layer for cnn and vit models
    cnn_former_params = [p for name, p in cnn_model.named_parameters() if 'fc' not in name]
    vit_former_params = [p for name, p in vit_model.named_parameters() if 'classifier' not in name]
    
    # define optimizer
    cnn_optimizer = optim.SGD([
            {'params': cnn_former_params, 'lr': ft_lr, 'weight_decay': weight_decay},
            {'params': cnn_model.fc.parameters(), 'lr': fc_lr, 'weight_decay': weight_decay}
        ], momentum=momentum
    )
    vit_optimizer = optim.SGD([
            {'params': vit_former_params, 'lr': ft_lr, 'weight_decay': weight_decay},
            {'params': vit_model.classifier.parameters(), 'lr': fc_lr, 'weight_decay': weight_decay}
        ], momentum=momentum
    )
    
    # define scheduler for learning rate decay
    def custom_step_scheduler(optimizer: optim, epoch: int, step_size: int, gamma: float):
        """
        Decay the learning rate of the parameter group by gamma every `step_size` epochs.
        """
        if epoch % step_size == 0 and epoch > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= gamma
                
    
    
    