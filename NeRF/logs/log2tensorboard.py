import os
import re
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from typing import List, Tuple

def write_loss_psnr(file: str):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    pattern = re.compile(r"\[TRAIN\] Iter: (\d+) Loss: ([\d\.]+)  PSNR: ([\d\.]+)")
    for line in lines:
        line = pattern.match(line)
        if line:
            iter_num = int(line.group(1))
            loss = float(line.group(2))
            psnr = float(line.group(3))
        
            writer.add_scalar('Trainin Loss', loss, iter_num)
            writer.add_scalar('PSNR', psnr, iter_num)
    


if __name__ == '__main__':
    
    writer = SummaryWriter(log_dir='./tensorboard')
    
    write_loss_psnr('./tiger.txt')
    
    writer.close()