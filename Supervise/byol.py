import torch
import torch.nn as nn
from typing import Tuple

class ProjectionHead(nn.Module):
    def __init__(self, in_dim :int, out_dim :int):
        super(ProjectionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x :torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class PredictionHead(nn.Module):
    def __init__(self, in_dim :int, out_dim :int):
        super(PredictionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x :torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class BYOL(nn.Module):
    """
    Bootstrap Your Own Latent, often abbrred as BYOL. It relies on two neural networks, referred to 
    as online and target networks, that interact and learn from each other.
    """
    def __init__(self, base_encoder :nn.Module, projection_dim :int, prediction_dim):
        super(BYOL, self).__init__()
        
        self.online_encoder = nn.Sequential(
            base_encoder,
            ProjectionHead(base_encoder.fc.in_features, projection_dim)
        )
        
        self.target_encoder = nn.Sequential(
            base_encoder,
            ProjectionHead(base_encoder.fc.in_features, projection_dim)
        )
        
        self.predictor = PredictionHead(projection_dim, prediction_dim)

        # initialize target network with the same weights as the online network
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        
        # freeze target network parameters
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def forward(self, x1 :torch.Tensor, x2 :torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # compute online projections
        online_proj1 = self.online_encoder(x1)
        online_proj2 = self.online_encoder(x2)

        # compute target projections
        with torch.no_grad():
            target_proj1 = self.target_encoder(x1)
            target_proj2 = self.target_encoder(x2)

        # compute predictions
        pred1 = self.predictor(online_proj1)
        pred2 = self.predictor(online_proj2)

        return pred1, pred2, target_proj1, target_proj2
    
    @staticmethod
    def loss(pred1 :torch.Tensor, pred2 :torch.Tensor, target1 :torch.Tensor, target2 :torch.Tensor):
        """
        Compute the loss between the predicted tensor and target tensor. Loss is summed over two 
        different views using `MSELoss`.
        ```python
        # NOTE: This is an abstract method, so it should be called with a reference to the class itself.
        >>> BYOL.loss(pred1, pred2, target1, target2)
        ```
        """
        loss = nn.MSELoss()
        return loss(pred1, target1) + loss(pred2, target2)