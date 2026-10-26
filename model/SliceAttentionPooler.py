
import torch
from torch import nn
from configs import get_config
cfg = get_config()

# ─────────────────────────────────────────────
#      Within-modality slice attention
# ─────────────────────────────────────────────
class SliceAttentionPooler(nn.Module):
    """
    Input:  (B, N_m, d)  + optional mask (B, N_m)
    Output: (B, d)        — one vector per modality per case
    """
    def __init__(self, d: int = 256, nhead: int = 4,
                 num_layers: int = 2, max_slices: int = cfg.glob.NUM_SLICES):
        super().__init__()
        self.pos_emb   = nn.Embedding(max_slices, d)
        # cls_token : a vector with size d that concat to slices of each modality ==>  shape changes from (B, N, d) to (B, N + 1, d)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d)) 
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=d * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, key_padding_mask=None):
        B, N, d = x.shape
        # add positional encoding
        pos = torch.arange(N, device=x.device) 
        x = x + self.pos_emb(pos)
        # Copies the CLS token for every item in the batch (1 cls per modality)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if key_padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)
            
        x = self.transformer(x, src_key_padding_mask=key_padding_mask) 
        return x[:, 0] 

