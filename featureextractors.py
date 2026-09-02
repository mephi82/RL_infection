import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision.models import resnet18, resnet50
import timm
import gymnasium as gym

class DictIdentityExtractor(BaseFeaturesExtractor):
    """
    Dict 관측을 '특징 벡터'로 바꾸지 않고, SB3가 요구하는 최소 텐서만 반환.
    - features_dim: 1 (형식상)
    - forward: 배치 크기에 맞는 zero 텐서 반환
    """
    def __init__(self, observation_space, features_dim: int = 1):
        super().__init__(observation_space, features_dim)
        self._feat_dim = features_dim

    def forward(self, observations):
        # observations는 Dict. 우리는 policy.forward에서 Dict를 직접 사용하므로,
        # 여기서는 SB3용 placeholder만 반환
        if isinstance(observations, dict):
            # 대표 키 하나로 배치 크기 추출
            for v in observations.values():
                B = v.shape[0]
                device = v.device
                break
        else:
            B = observations.shape[0]
            device = observations.device
        return torch.zeros(B, self._feat_dim, device=device)

class SwinExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        in_channels = observation_space.shape[0]
        image_size = observation_space.shape[1:]  # (H, W)

        # Swin Transformer 모델 로드 (timm 이용)
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            features_only=True,
            in_chans=in_channels
        )

        # 출력 shape 확인용 forward pass
        with torch.no_grad():
            sample = torch.randn(1, in_channels, *image_size)
            feats = self.backbone(sample)[-1]  # 가장 마지막 단계 사용
            flatten_dim = feats.shape[1] * feats.shape[2] * feats.shape[3]

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(flatten_dim, features_dim)

    def forward(self, x):
        feats = self.backbone(x)[-1]   # Swin의 마지막 stage feature map
        x = self.flatten(feats)
        return self.linear(x)

class ResNet50Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        in_channels = observation_space.shape[0]
        resnet = resnet50()

        # 입력 채널이 1이면 conv1 수정
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # fc 제거 (avgpool까지 사용)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # (B, 2048, 1, 1)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(2048, features_dim)

    def forward(self, observations):
        x = self.resnet(observations)  # (B, 2048, 1, 1)
        x = self.flatten(x)            # (B, 2048)
        return self.linear(x)          # (B, features_dim)
    
class ResNet18Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        # 입력 채널 수 확인 (예: 1채널 히트맵이면 수정 필요)
        in_channels = observation_space.shape[0]

        # torchvision의 resnet18 로드
        # self.resnet = resnet18(pretrained=True)
        self.resnet = resnet18()

        # 1채널 입력일 경우 첫 conv layer 수정
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 마지막 fc 제거 → 특징 벡터만 추출
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # (B, 512, 1, 1)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, features_dim)

    def forward(self, observations):
        x = self.resnet(observations)  # (B, 512, 1, 1)
        x = self.flatten(x)            # (B, 512)
        return self.linear(x)
        
class ResNet18ExtractorWithTime(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim=features_dim+1)

        # 입력 채널 수 확인 (예: 1채널 히트맵이면 수정 필요)
        image_space = observation_space.spaces['image']
        in_channels = image_space.shape[0]

        # torchvision의 resnet18 로드
        # self.resnet = resnet18(pretrained=True)
        self.resnet = resnet18()

        # 1채널 입력일 경우 첫 conv layer 수정
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 마지막 fc 제거 → 특징 벡터만 추출
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # (B, 512, 1, 1)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, features_dim)

    def forward(self, observations):
        x = self.resnet(observations['image'])  # (B, 512, 1, 1)
        x = self.flatten(x)            # (B, 512)
        return torch.cat([self.linear(x), observations['time'].float()], dim=1)  # (B, features_dim + 1)

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

class ImageTransformerExtractor(BaseFeaturesExtractor):
    """
    ViT-style feature extractor for (C, H, W) image observations.
    - Splits image into non-overlapping patches
    - Linear embeds patches + learnable positional embeddings
    - Transformer encoder -> pooled -> FC -> features_dim
    """
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        patch_size: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        pool: str = "mean",   # "mean" or "cls"
    ):
        super().__init__(observation_space, features_dim)

        assert len(observation_space.shape) == 3, "Expect image obs with shape (C,H,W)"
        C, H, W = observation_space.shape
        assert H % patch_size == 0 and W % patch_size == 0, \
            f"H ({H}) and W ({W}) must be divisible by patch_size ({patch_size})"

        self.C, self.H, self.W = C, H, W
        self.patch_size = patch_size
        self.num_patches_h = H // patch_size
        self.num_patches_w = W // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        patch_dim = C * patch_size * patch_size
        self.pool = pool

        # Optional [CLS] token for pooling
        self.use_cls = (pool == "cls")
        cls_tokens = 1 if self.use_cls else 0

        # Patch embedding (flattened patch -> d_model)
        self.patch_embed = nn.Linear(patch_dim, d_model)

        # Positional embedding (learnable)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + cls_tokens, d_model))

        # Optional learnable CLS token
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.register_parameter("cls_token", None)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, features_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) -> (B, num_patches, patch_dim)
        Uses F.unfold for speed.
        """
        B, C, H, W = x.shape
        p = self.patch_size
        # (B, C*p*p, L) where L = num_patches
        patches = F.unfold(x, kernel_size=p, stride=p)          # (B, patch_dim, L)
        patches = patches.transpose(1, 2)                        # (B, L, patch_dim)
        return patches

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, C, H, W)
        patches = self._to_patches(obs)                          # (B, L, patch_dim)
        x = self.patch_embed(patches)                            # (B, L, d_model)

        if self.use_cls:
            cls = self.cls_token.expand(x.size(0), -1, -1)       # (B, 1, d_model)
            x = torch.cat([cls, x], dim=1)                       # (B, 1+L, d_model)

        x = x + self.pos_embedding[:, : x.size(1), :]            # (B, 1+L or L, d_model)
        x = self.transformer(x)                                  # (B, seq, d_model)
        x = self.norm(x)

        if self.use_cls:
            x = x[:, 0]                                          # (B, d_model)
        else:
            x = x.mean(dim=1)                                    # (B, d_model)

        return self.fc(x)                                        # (B, features_dim)        