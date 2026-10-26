import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
import torch
from torch.cuda.amp import autocast, GradScaler

from dataset.SafeSubmissionDataset import SafeSubmissionDataset
from train.Trainer import TrainerSingleFold
from utils.utils import (start_mem_watcher, safe_collate)
from XAI.xai import run_xai
from model.MRI25DFusionModelViewRoutingDualEncoder import MRI25DFusionModelViewRoutingDualEncoder
from configs import get_config
cfg = get_config()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer=TrainerSingleFold(train_df= cfg.data.TRAIN_LABELD ,epochs = (cfg.data.NUM_CYCLE * cfg.glob.NUM_CHUNKS), device = device, scaler = GradScaler() )
start_mem_watcher()
trainer.meta_trainer(drop_noise=False, verbose=True)
trainer.training()

# ------XAI

combined_model = MRI25DFusionModelViewRoutingDualEncoder(
            d=256, nhead=4, num_layers=2,
            num_modalities=3, max_slices=cfg.glob.NUM_SLICES,
            num_metadata_features=33, d_meta=32
        )
path = f"{cfg.data.save_chkp}/best_model_checkpoint.pth"
ckpt = torch.load(path, map_location='cpu', weights_only=False) 
combined_model.load_state_dict(ckpt['model'])

combined_model = combined_model.to('cuda').eval()

gt_df = cfg.data.GT_VAL.copy()
ds     = SafeSubmissionDataset(gt_df, cfg.data.LIGHT_DF, meta_ref=None)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=safe_collate)

mean_plane = run_xai(combined_model, loader, gt_df, out_dir='xai_out', n_study_figs=5)