import re
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, List

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

    def get_loss_and_acc(file_path: str) -> Tuple[List, List, List, List]:
        # initialize epoch counter
        epoch = 1
        
        # store the losses and accuracies
        train_losses, test_losses, top1_accs, top5_accs = [], [], [], []
        
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
                    top5_acc = float(test_loss_match.group(2))
                    top1_acc = float(test_loss_match.group(3))
                    test_losses.append(test_loss)
                    top1_accs.append(top1_acc)
                    top5_accs.append(top5_acc)
                    epoch += 1

        return train_losses, test_losses, top1_accs, top5_accs
        
    # process both log files
    no_train_loss, no_test_loss, no_top1_acc, no_top5_acc = get_loss_and_acc(no_cutmix_path)
    with_train_loss, with_test_loss, with_top1_acc, with_top5_acc = get_loss_and_acc(with_cutmix_path)
    
    # write them to tensorboard
    for i in range(len(no_train_loss)):
        writer.add_scalars('Loss', {
            '/No_CutMix/Train_Loss': no_train_loss[i],
            '/No_CutMix/Test_Loss': no_test_loss[i],
            '/With_CutMix/Train_Loss': with_train_loss[i],
            '/With_CutMix/Test_Loss': with_test_loss[i],
        }, i + 1)
        
        writer.add_scalars('Accuracy', {
            '/No_CutMix/Top1_Acc': no_top1_acc[i],
            '/No_CutMix/Top5_Acc': no_top5_acc[i],
            '/With_CutMix/Top1_Acc': with_top1_acc[i],
            '/With_CutMix/Top5_Acc': with_top5_acc[i],
        }, i + 1)

    # close the TensorBoard SummaryWriter
    writer.close()
    
    
if __name__ == '__main__':
    
    file_paths = [
        ('./CNN/15--0.0005--0.01--64--0.log', './CNN/15--0.0005--0.01--64--1.log'),
        ('./ViT/15--0.0005--0.01--64--0.log', './ViT/15--0.0005--0.01--64--1.log')
    ]

    board_paths = [
        './tensorboard/CNN',
        './tensorboard/ViT'
    ]

    for file_path, board_path in zip(file_paths, board_paths):
        convert(file_path, board_path)