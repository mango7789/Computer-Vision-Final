import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from copy import deepcopy
from typing import Tuple

class Encoder(nn.Module):
    """
    Encoder of the image, here it's choosen as `ResNet-18`. The parameter `pretrain` can be turned on or
    off to use pretrained `ResNet-18` or not.
    """
    def __init__(self, pretrain: bool=False):
        super(Encoder, self).__init__()
        if pretrain:
            base_model = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
        else:
            base_model = models.resnet18(weights=None)
        # exclude the last fc layer
        self.encoder = nn.Sequential(
            *list(base_model.children())[:-1]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encode images and flatten each corresponding feature in a mini-batch
        return self.encoder(x).flatten(1)


class MLP(nn.Module):
    """
    MLP, Multi-Layer Perceptron, used as the projection head and prediction head.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class BYOL(nn.Module):
    """
    Bootstrap Your Own Latent, often abbrred as BYOL. It relies on two neural networks, referred to 
    as online and target networks, that interact and learn from each other.
    """
    def __init__(self, base_encoder, hidden_dim :int, output_dim :int, momentum :float):
        super(BYOL, self).__init__()
        self.momentum = momentum

        # online network
        self.online_encoder = base_encoder()
        self.online_projection = MLP(512, hidden_dim, output_dim)
        self.online_prediction = MLP(output_dim, hidden_dim, output_dim)

        # target network, use the same initialized weights as online network 
        self.target_encoder = deepcopy(self.online_encoder)
        self.target_projection = deepcopy(self.online_projection)

        # freeze the target network parameters
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projection.parameters():
            param.requires_grad = False


    def forward(self, x1 :torch.Tensor, x2 :torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # online network forward pass
        online_proj1 = self.online_projection(self.online_encoder(x1))
        online_proj2 = self.online_projection(self.online_encoder(x2))

        online_pred1 = self.online_prediction(online_proj1)
        online_pred2 = self.online_prediction(online_proj2)

        # target network forward pass
        with torch.no_grad():
            target_proj1 = self.target_projection(self.target_encoder(x1))
            target_proj2 = self.target_projection(self.target_encoder(x2))

        return online_pred1, online_pred2, target_proj1, target_proj2


    @torch.no_grad()
    def update_target_network(self):
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = self.momentum * target_params.data + (1 - self.momentum) * online_params.data

        for online_params, target_params in zip(self.online_projection.parameters(), self.target_projection.parameters()):
            target_params.data = self.momentum * target_params.data + (1 - self.momentum) * online_params.data
    
    
    @staticmethod
    def loss(pred1 :torch.Tensor, pred2 :torch.Tensor, target1 :torch.Tensor, target2 :torch.Tensor):
        """
        Compute the loss between the predicted tensor and target tensor. Loss is summed over two 
        different views using `cosine_similarity`.
        ```python
        # NOTE: This is an abstract method, so it should be called with a reference to the class itself.
        >>> BYOL.loss(pred1, pred2, target1, target2)
        ```
        """
        loss = 2 - 2 * (
            F.cosine_similarity(pred1, target2.detach(), dim=-1).mean() +
            F.cosine_similarity(pred2, target1.detach(), dim=-1).mean()
        )
        return loss