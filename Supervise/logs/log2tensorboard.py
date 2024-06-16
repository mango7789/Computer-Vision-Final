import os
import re
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from typing import List, Tuple

def write_train_loss(root: str = './BYOL'):
    dir_list = os.listdir(root)
    train_loss_pattern = re.compile(r'Training loss is (\d+\.\d+)')
    
    loss_dict = defaultdict(list)
    
    def get_single_byol(full_path: str) -> List:
        train_losses = []
        with open(full_path, 'r') as f:
            for line in f:
                train_loss_match = train_loss_pattern.search(line)
                if train_loss_match:
                    train_loss = float(train_loss_match.group(1))
                    train_losses.append(train_loss)
        return train_losses

    for dir_ in dir_list:
        full_path = os.path.join(root, dir_)
        loss_dict[dir_] = get_single_byol(full_path)
        
    for i in range(len(loss_dict[dir_list[0]])):
        writer.add_scalars('Encoder/Training Loss', {
            config[:-4]: losses[i] for config, losses in loss_dict.items()
        }, i + 1)
   
def write_linear_acc(root: str = './Linear-Classifier'):
    train_loss_pattern = re.compile(r'Training loss is (\d+\.\d+)')
    test_loss_pattern = re.compile(r'Validation loss is (\d+\.\d+), Validation accuracy is (\d+\.\d+)')
        
    def get_loss_and_acc(file_path: str) -> Tuple[List, List, List]:
        # store the losses and accuracies
        train_losses, test_losses, test_accs = [], [], []
        
        # read and parse the log file
        with open(file_path, 'r') as log_file:
            for line in log_file:
                train_loss_match = train_loss_pattern.search(line)
                if train_loss_match:
                    train_loss = float(train_loss_match.group(1))
                    train_losses.append(train_loss)

                test_loss_match = test_loss_pattern.search(line)
                if test_loss_match:
                    test_loss = float(test_loss_match.group(1))
                    test_acc = float(test_loss_match.group(2))
                    test_losses.append(test_loss)
                    test_accs.append(test_acc)
        
        return train_losses, test_losses, test_accs
    
    dir_list = os.listdir(root)
    for dir_ in dir_list:
        train_loss, test_loss, test_acc = get_loss_and_acc(os.path.join(root, dir_))
        
        for i in range(len(train_loss)):
            model_type = dir_.split('.')[0]
            
            writer.add_scalars('Linear Classifier/Loss', {
                 f'{model_type}/Train Loss': train_loss[i],
                 f'{model_type}/Validation Loss': test_loss[i]
            }, i + 1)

            writer.add_scalars('Linear Classifier/Accuracy', {
                 f'{model_type}/Validation Accuracy': test_acc[i],
            }, i + 1)
    
def write_resnet_loss(root: str = './ResNet-18'):
    dir_list = os.listdir(root)
    train_loss_pattern = re.compile(r'Training loss is (\d+\.\d+)')
        
    def get_single_byol(full_path: str) -> List:
        train_losses = []
        with open(full_path, 'r') as f:
            for line in f:
                train_loss_match = train_loss_pattern.search(line)
                if train_loss_match:
                    train_loss = float(train_loss_match.group(1))
                    train_losses.append(train_loss)
        return train_losses

    losses = get_single_byol(os.path.join(root, dir_list[0]))
    
    for i in range(20):
        writer.add_scalars('ResNet/Training Loss', {
            'Pretrain': losses[i],
            'Random Init': losses[20 + i],
        }, i + 1)


if __name__ == '__main__':
    
    writer = SummaryWriter(log_dir='./tensorboard')
    
    write_train_loss()
    write_linear_acc()
    write_resnet_loss()
    
    writer.close()