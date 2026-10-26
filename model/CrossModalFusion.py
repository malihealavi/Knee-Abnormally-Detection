import torch
from torch import nn
from configs import get_config
cfg = get_config()

# ─────────────────────────────────────────────
#  Cross-modality fusion attention
# ─────────────────────────────────────────────
class CrossModalFusion(nn.Module):
    """
    Input:  list of M tensors each (B, d) | for each Modality we have B vectors with size d
    Output: (B, d)  — fused across all modalities | all Modalities merge together so One vector per case
    """
    def __init__(self, d: int = 256, nhead: int = 4,
                 num_layers: int = 2, num_modalities: int = 3):
        super().__init__()
        self.mod_emb = nn.Embedding(num_modalities, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=d * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)

    def forward(self, modality_vecs):
        # modality_vecs: list of 4 tensors each (B, d)
        x   = torch.stack(modality_vecs, dim=1)         # (B, 4, d)
        ids = torch.arange(x.size(1), device=x.device)
        x   = x + self.mod_emb(ids)                     # (B, 4, d)
        x   = self.transformer(x)                        # (B, 4, d)
        return x.mean(dim=1)                             # (B, d)

