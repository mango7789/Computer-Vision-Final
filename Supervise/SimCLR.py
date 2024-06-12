import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

class ResNetSimCLR(nn.Module):
    """
    A Simple Framework for Contrastive Learning of Visual Representations with `base_encoder`
    as resnet18. 
    """
    def __init__(self, base_encoder: nn.Module):
        super(ResNetSimCLR, self).__init__()
        
        self.backbone = base_encoder
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @staticmethod
    def info_nce_loss(batch_size: int, features: torch.Tensor, device: torch.device, temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:

        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / temperature
        return logits, labels