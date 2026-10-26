import torch
from torch import nn
import timm
from transformers import  AutoModel

from configs import get_config
cfg = get_config()

# ─────────────────────────────────────────────
# Shared 2D CNN slice encoder
# ─────────────────────────────────────────────
class SliceEncoder(nn.Module):
    """
    Input:  (B*N_m, 3, H, W)   — flat batch of triple-stacked slices
    Output: (B*N_m, d)          — one feature vector per slice
    """
    def __init__(self, d: int = 256, bb: str = None):
        super().__init__()
        bb = bb if bb is not None else ENCODER_BB   # read at call time
        self.bb = bb
        self.proj = None

        if bb == 'convnext_nano':
            backbone = timm.create_model("convnext_nano", pretrained=False, num_classes=0)
            loaded = torch.load(
                '/kaggle/input/datasets/malihealavi/knee-pretrained-encoder/ssl_convnext_nano.pth',
                #/kaggle/input/datasets/malihealavi/knee-pretrained-encoder/ssl_convnext_nano.pth
                map_location="cpu",
                weights_only=False,
            )
            sd = loaded.state_dict() if hasattr(loaded, "state_dict") else loaded
            sd = {k: v for k, v in sd.items() if not k.startswith("head.fc")}
            missing, unexpected = backbone.load_state_dict(sd, strict=False)
            assert not unexpected, f"Unexpected keys — architecture mismatch: {unexpected}"
            print(f'Selected Encoder: {bb}')
            # --- Count and print parameters ---
            total_params = sum(p.numel() for p in backbone.parameters())
            trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            print(f"Total Parameters: {total_params:,}")
            print(f"Trainable Parameters: {trainable_params:,}")
            # ----------------------------------
            feature = backbone.num_features        # e.g. 640 for convnext_nano
            self.proj = nn.Linear(feature, d)
            for p in backbone.parameters():         # freeze backbone
                p.requires_grad = False
            trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            print(f"Trainable Parameters: {trainable_params:,}")
        elif bb == 'dinov2_small':
            DINOV2_LOCAL_PATH = "/kaggle/input/datasets/malihealavi/knee-pretrained-encoder/ssl_dinov2_small"
            # DINOV2_LOCAL_PATH = "/kaggle/input/rsna-knee-model/dinov2_knee_ssl_best_probe"
            backbone = AutoModel.from_pretrained(DINOV2_LOCAL_PATH, local_files_only=True)
            feature = backbone.config.hidden_size          # 768 for dinov2-base
            self.proj = nn.Linear(feature, d)                # dino has no head to swap, so project manually
            print(f'Selected Encoder: {bb}')
            for p in backbone.parameters():
                p.requires_grad = False

        elif bb == 'dinov3_small':
            DINOV2_LOCAL_PATH = "/kaggle/input/models/metaresearch/dinov2/pytorch/small/1"
            backbone = AutoModel.from_pretrained(DINOV2_LOCAL_PATH, local_files_only=True)
            feature = backbone.config.hidden_size          # 768 for dinov2-base
            self.proj = nn.Linear(feature, d)                # dino has no head to swap, so project manually
            print(f'Selected Encoder: {bb}')
            for p in backbone.parameters():
                p.requires_grad = False

        else:
            raise ValueError(f"Unknown backbone: {bb}")

        self.backbone = backbone

    def forward(self, x):
        if self.bb == 'dinov2_small':
            outputs = self.backbone(x)
            x = outputs.last_hidden_state[:, 0, :]   # CLS token, (B*N_m, hidden)
            x = self.proj(x)                          # -> (B*N_m, d)
        else:
            x = self.backbone(x)                       # resnet: (B*N_m, d) already
            if self.proj is not None:                   # convnext: (B*N_m, feature) -> proj -> (B*N_m, d)
                x = self.proj(x)
        return x 
