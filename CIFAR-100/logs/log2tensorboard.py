import os
import re
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple

def convert(file_paths: Tuple[str, str], tensorboard_path: str):
    """
    Extract the loss and accuracy from two log files and write them into TensorBoard.
    """
    
    no_cutmix_path, with_cutmix_path = file_paths

    # initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=tensorboard_path)

    # regular expressions to match the relevant log lines
    train_loss_pattern = re.compile(r'Training loss is (\d+\.\d+)')
    test_loss_pattern = re.compile(r'Testing loss is (\d+\.\d+), Top-5 accuracy is (\d+\.\d+), Top-1 accuracy is (\d+\.\d+)')

    def write_to_tensorboard(file_path: str, label: str):
        # initialize epoch counter
        epoch = 1
        
        # read and parse the log file
        with open(file_path, 'r') as log_file:
            for line in log_file:
                train_loss_match = train_loss_pattern.search(line)
                if train_loss_match:
                    train_loss = float(train_loss_match.group(1))
                    writer.add_scalar(f'{label}/Training Loss', train_loss, epoch)

                test_loss_match = test_loss_pattern.search(line)
                if test_loss_match:
                    test_loss = float(test_loss_match.group(1))
                    top5_acc = float(test_loss_match.group(2))
                    top1_acc = float(test_loss_match.group(3))
                    writer.add_scalar(f'{label}/Testing Loss', test_loss, epoch)
                    writer.add_scalar(f'{label}/Top-5 Accuracy', top5_acc, epoch)
                    writer.add_scalar(f'{label}/Top-1 Accuracy', top1_acc, epoch)
                    epoch += 1

    # process both log files
    write_to_tensorboard(no_cutmix_path, 'No CutMix')
    write_to_tensorboard(with_cutmix_path, 'With CutMix')

    # close the TensorBoard SummaryWriter
    writer.close()
    
    
if __name__ == '__main__':
    
    file_paths = [
        ('./CNN/15--0.0005--0.01--64--0.log', './CNN/15--0.0005--0.01--64--1.log'),
        ('./ViT/15--0.0005--0.01--64--0.log', './ViT/15--0.0005--0.01--64--1.log')
    ]

    board_paths = [
        './CNN/tensorboard',
        './ViT/tensorboard'
    ]

    for file_path, board_path in zip(file_paths, board_paths):
        convert(file_path, board_path)