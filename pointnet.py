import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class PointNetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        # input shape: (B, N, 3)
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),  # point-wise feature
            nn.ReLU()
        )

    def forward(self, observations):
        # observations: (B, N, 3)
        x = self.point_mlp(observations)         # (B, N, D)
        x = torch.max(x, dim=1).values           # (B, D) — MaxPool over points
        return x

class TransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, nhead=4, num_layers=2):
        super().__init__(observation_space, features_dim)

        self.input_dim = observation_space.shape[-1]  # ex: 3 (x, y, attr)
        self.seq_len = observation_space.shape[0]     # N

        self.embedding = nn.Linear(self.input_dim, 64)

        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(64, features_dim)

    def forward(self, obs):
        # obs: (B, N, 3)
        x = self.embedding(obs)              # (B, N, 64)
        x = self.transformer(x)              # (B, N, 64)
        x = x.max(dim=1).values              # (B, 64)
        return self.fc(x)                    # (B, features_dim)        