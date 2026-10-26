
import torch
from torch import nn

from model.CrossModalFusion import CrossModalFusion
from model.SliceAttentionPooler import SliceAttentionPooler
from model.SliceEncoder import SliceEncoder
from configs import get_config
cfg = get_config()


class MRI25DFusionModelViewRoutingDualEncoder(nn.Module):
    def __init__(self,
                 d: int = 256,
                 nhead: int = 4,
                 num_layers: int = 2,
                 num_modalities: int = 3,
                 max_slices: int = cfg.glob.NUM_SLICES,
                 num_metadata_features : int= 33,
                 d_meta: int = 32):
        super().__init__()
        self.d = d 
        self.dino_slice_encoder = SliceEncoder(d=d, bb='dinov2_small')
        self.conv_slice_encoder = SliceEncoder(d=d, bb='convnext_nano')
        self.slice_fuse = nn.Sequential(
                    nn.Linear(d * 2, d),
                    nn.LayerNorm(d),
                )
        self.slice_poolers = nn.ModuleList([
            SliceAttentionPooler(d=d, nhead=nhead,
                                 num_layers=num_layers,
                                 max_slices=max_slices)
            for _ in range(num_modalities)
        ])

        self.fusion = CrossModalFusion(d=d, nhead=nhead,
                                       num_layers=num_layers,
                                       num_modalities=num_modalities)

        # ── Target-specific view routing ────────────────────────────
        # Order below MUST match your true label column order.
        # Check against LIGHT_DF and fix if different.
        self.LABEL_ORDER = [
            "acl", "mcl", "medial_meniscus", "lateral_meniscus",
            "medial_oa", "lateral_oa", "pf_oa", "effusion", "synovitis",
            "bakers_cyst", "contusion", "fracture",
        ]
        AXIAL_LABELS   = ["pf_oa", "effusion", "synovitis"]
        SC_LABELS      = ["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa", "lateral_oa"]
        GENERAL_LABELS = ["bakers_cyst", "contusion", "fracture"]
        grouped = AXIAL_LABELS + SC_LABELS + GENERAL_LABELS
        perm = [grouped.index(l) for l in self.LABEL_ORDER]
        self.register_buffer("label_perm", torch.tensor(perm, dtype=torch.long))

        def make_head(out_dim):
            return nn.Sequential(
                nn.Linear(d*2, d ),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(d , out_dim),
            )

        self.axial_head   = make_head(len(AXIAL_LABELS))    # PF OA, Effusion, Synovitis
        self.sc_head      = make_head(len(SC_LABELS))       # ACL, MCL, Menisci, Medial/Lateral OA
        self.general_head = make_head(len(GENERAL_LABELS))  # Baker's cyst, Contusion, Fracture
        # self.metadata_encoder = MetadataEncoder(num_features= num_metadata_features, d=d_meta)
        # self.meta_fuse = nn.Linear(d + d_meta, d)

    def forward(self, modalities, meta=None, masks=None):
        if masks is None:
            masks = [None] * len(modalities)
 
        modality_vecs = []
        for slices, pooler, mask in zip(modalities, self.slice_poolers, masks):
            B, N_m, C, H, W = slices.shape
            flat = slices.reshape(B * N_m, C, H, W) 
            feats_dino = self.dino_slice_encoder(flat).view(B, N_m, -1)
            feats_conv = self.conv_slice_encoder(flat).view(B, N_m, -1)
            assert feats_dino.shape[-1] == self.d and feats_conv.shape[-1] == self.d
            
            feats_cat = self.slice_fuse(torch.cat([feats_dino, feats_conv], dim=-1))
            pooled = pooler(feats_cat, mask)
            modality_vecs.append(pooled)
 
        fused   = self.fusion(modality_vecs)
        axial   = modality_vecs[0]
        sc      = (modality_vecs[1] + modality_vecs[2]) / 2
        all_avg = torch.stack(modality_vecs, dim=1).mean(dim=1)      # (B, d)

        out = torch.cat([self.axial_head(torch.cat([axial, fused], dim=1)),
                         self.sc_head(torch.cat([sc, fused], dim=1)),
                         self.general_head(torch.cat([all_avg, fused], dim=1)),
                        ], dim=1)
        return out[:, self.label_perm]

