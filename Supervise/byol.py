import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ProjectionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)

class PredictionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PredictionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)

class BYOL(nn.Module):
    def __init__(self, base_encoder, projection_dim=256, prediction_dim=4096):
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

        # Initialize target network with the same weights as the online network
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        
        # Freeze target network parameters
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        # Compute online projections
        online_proj1 = self.online_encoder(x1)
        online_proj2 = self.online_encoder(x2)

        # Compute target projections
        with torch.no_grad():
            target_proj1 = self.target_encoder(x1)
            target_proj2 = self.target_encoder(x2)

        # Compute predictions
        pred1 = self.predictor(online_proj1)
        pred2 = self.predictor(online_proj2)

        return pred1, pred2, target_proj1, target_proj2