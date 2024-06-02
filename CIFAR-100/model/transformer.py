import torch
import torch.nn as nn


class SimpleTransformer(nn.Module):
    def __init__(
        self, 
        num_classes :int=100, 
        emb_dim :int=128, 
        heads :int=8, 
        depth :int=6, 
        mlp_dim :int=256, 
        dropout :float=0.1
    ):
        super(SimpleTransformer, self).__init__()
        self.conv1 = nn.Conv2d(3, emb_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.transformer = nn.Transformer(emb_dim, heads, depth, depth, mlp_dim, dropout, batch_first=True)
        self.fc = nn.Linear(emb_dim * 16 * 16, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1, x.size(1))
        x = self.transformer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x