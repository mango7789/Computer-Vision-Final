import re
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple

def get_loss_psnr(file: str) -> Tuple[List, List]:
    losses = []
    psnrs = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    pattern = re.compile(r"\[TRAIN\] Iter: (\d+) Loss: ([\d\.]+)  PSNR: ([\d\.]+)")
    for line in lines:
        line = pattern.match(line)
        if line:
            loss = float(line.group(2))
            psnr = float(line.group(3))
        
            losses.append(loss)
            psnrs.append(psnr)
    
    return losses, psnrs


def write_loss_psnr(self_res: str, offc_res: str):
    self_loss, self_psnr = get_loss_psnr(self_res)
    offc_loss, offc_psnr = get_loss_psnr(offc_res)
    
    for i in range(len(self_loss)):
        writer.add_scalars('Training Loss', {
            'hhsw': self_loss[i],
            'vasedeck': offc_loss[i]
        }, (i + 1) * 100)
        writer.add_scalars('PSNR', {
            'hhsw': self_psnr[i],
            'vasedeck': offc_psnr[i]
        }, (i + 1) * 100)


if __name__ == '__main__':
    
    writer = SummaryWriter(log_dir='./tensorboard')
    
    write_loss_psnr('./hhsw.txt', './vasedeck.txt')
    
    writer.close()