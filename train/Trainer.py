import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import numpy as np
import re
import gc
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

 
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import  CosineAnnealingLR

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
            
import re

  
from dataset.KneeDataset import KneeDataset
from model.MRI25DFusionModelViewRoutingDualEncoder import MRI25DFusionModelViewRoutingDualEncoder
from utils.utils import (
    blend,
    cleanup,
    fit_label_weights,
    macro_auc,
    plot_loss,
    search_blend_weight,
    unwrap,
    mri_collate_fn,
    print_label_weights
)
from configs import get_config
cfg = get_config()

class TrainerSingleFold():
    def __init__(self,
                 train_df: pd.DataFrame,
                 epochs: int,
                 device: str,
                 scaler: Optional[torch.amp.GradScaler] = None):  # FIX WARN 1

        tmp_ds = KneeDataset(train_labeled=train_df.head(1), series_merged=cfg.data.LIGHT_DF,
                     split='train', augmentation=False)
        num_meta = tmp_ds.num_meta
        print("num metadata features:", num_meta)
        
        self.model = MRI25DFusionModelViewRoutingDualEncoder(
            d=cfg.model.d, nhead=cfg.model.nhead, num_layers=cfg.model.num_layers,
            num_modalities=cfg.model.num_modalities, max_slices=cfg.glob.NUM_SLICES,
            num_metadata_features=num_meta, d_meta=cfg.model.d_meta
        )

        
        self.model.to(device)
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs")
        #     self.model = nn.DataParallel(self.model)
        
        self.epochs    = epochs

        self.backbone_params = [p for n, p in self.model.named_parameters()
                             if "slice_encoder.backbone" in n]
        self.head_params = [p for n, p in self.model.named_parameters()
                             if "slice_encoder.backbone" not in n]
        
        self.optimizer = optim.AdamW([
            {"params": self.backbone_params, "lr": 1e-5},
            {"params": self.head_params,     "lr": 1e-4},
        ], weight_decay=5e-3)
        
        self.device    = device
        self.scaler    = scaler
        self.accumulation_step = cfg.train.ACCUMULATION_STEP
        
        self.train, val        = train_test_split(train_df, test_size=0.1, random_state=42)
        self.val   = val.reset_index(drop=True)
        self.GT_VAL   = cfg.data.GT_VAL.reset_index(drop=True) #58 Ground Truth samples
        

        self.val_ds   = KneeDataset(train_labeled = self.val , series_merged = cfg.data.LIGHT_DF, split='train', augmentation=False)
        self.gt_val_ds   = KneeDataset(train_labeled = self.GT_VAL , series_merged = cfg.data.LIGHT_DF, split='train', augmentation=False)

        self.criterion    = nn.BCEWithLogitsLoss()
        self.best_auc     = 0.0
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.train.NUM_CYCLE, eta_min=5e-7)
        self.current_cycle =None
        self.current_epoch = None
        self.train_loss =[]
        self.valid_loss =[]
        self.valid_auc=[]
        self.patience = 0
        self.stall = 0
        self.resume_cycle = 0
        self.resume_epoch = -1
        self.epoch_preds = {}
        self.blend_log = []
        ckpt_result = self.load_checkpoint()   # keep this call, but capture the return value
        if ckpt_result is not None:
            self.resume_cycle, self.resume_epoch = ckpt_result

    def _unfreeze_slice_encoder(self, cycle):
    
        if cycle < cfg.train.UNFREEZE_START_CYCLE:
            n_unfrozen = 0
        else:
            n_unfrozen = 1 + (cycle - cfg.train.UNFREEZE_START_CYCLE) // cfg.train.CYCLES_PER_BLOCK
    
        n_dino = min(n_unfrozen, cfg.train.MAX_UNFROZEN_BLOCKS)
        n_conv = min((n_dino + 2) // 3, cfg.train.TOTAL_CONV_STAGES)   # 1 conv stage per ~3 DINO blocks
    
        for n, p in self.model.named_parameters():
            if "dino_slice_encoder.backbone" in n:
                m = re.search(r"\.layer\.(\d+)\.", n)
                if m:
                    p.requires_grad = int(m.group(1)) >= cfg.train.TOTAL_BACKBONE_BLOCKS - n_dino
                elif "layernorm" in n:
                    p.requires_grad = n_dino > 0
                else:
                    p.requires_grad = False
    
            elif "conv_slice_encoder.backbone" in n:
                m = re.search(r"\.stages\.(\d+)\.", n)
                if m:
                    p.requires_grad = int(m.group(1)) >= cfg.train.TOTAL_CONV_STAGES - n_conv
                elif "head.norm" in n:                        # final norm
                    p.requires_grad = n_conv > 0
                else:
                    p.requires_grad = False
    
        # Batch size: keep effective batch = 128
        if cycle >= 4:
            cfg.train.BATCH_SIZE = min(cfg.train.BATCH_SIZE, 2)
        elif cycle >= 2:
            cfg.train.BATCH_SIZE = min(cfg.train.BATCH_SIZE, 4)
        cfg.train.ACCUMULATION_STEP = 128 // cfg.train.BATCH_SIZE
        self.accumulation_step = cfg.train.ACCUMULATION_STEP
        # NOTE: rebuild your DataLoader here if cfg.train.BATCH_SIZE changed
    
        print(f"[cycle {cycle}] dino: last {n_dino}/{cfg.train.TOTAL_BACKBONE_BLOCKS} layers | "
              f"conv: last {n_conv}/{cfg.train.TOTAL_CONV_STAGES} stages | "
              f"batch:{cfg.train.BATCH_SIZE}, acc:{cfg.train.ACCUMULATION_STEP}")


            
    def _save_checkpoint(self, path="checkpoint.pth"):
        ckpt = {
            'epoch': self.current_epoch,
            'cycle': self.current_cycle,
            'model': unwrap(self.model).state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_auc': self.best_auc,
            'patience': self.patience,
            'train_loss': self.train_loss,
            'val_loss': self.valid_loss,
            'valid_auc': self.valid_auc,
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path="./assets/checkpoint/last_checkpoint.pth"):
        
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}, starting fresh.")
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False) 
        unwrap(self.model).load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.best_auc = ckpt['best_auc']
        self.patience = ckpt['patience']
        self.train_loss = ckpt['train_loss']
        self.valid_loss = ckpt['val_loss']
        self.valid_auc = ckpt['valid_auc']
        print(f"Resume Previous chkpoit, cycle: {ckpt['cycle']}, epoch:{ckpt['epoch']}, patience:{self.patience}, best_auc = {self.best_auc}")
        self._unfreeze_slice_encoder(ckpt['cycle'])
            
        return ckpt['cycle'], ckpt['epoch']
        
    def _unpack(self, batch):
        modalities, meta, targets = batch
        modalities = [m.to(self.device, dtype=torch.float32) for m in modalities]
        meta    = meta.to(self.device, dtype=torch.float32)
        targets    = targets.to(self.device, dtype=torch.float32)
        return modalities, meta, targets
        


    def meta_trainer(self, drop_noise=False, verbose=True):
        """
        Metadata-only models, fitted on self.train, predicting self.val and self.GT_VAL.
        Output rows are in the SAME order as self.val / self.GT_VAL
        (so they line up with valid_loader / gt_valid_loader, which use shuffle=False).

        Saves:
          self.meta_preds[name] = {"val": (N_val, 12), "gt": (N_gt, 12)}   for LR / LGBM / HGB
          self.meta_name        = best model by macro AUC on self.val
          self.meta_val, self.meta_gt = predictions of that best model
          self.y_val_bin, self.y_gt_bin = binarized labels (N, 12)
        """
        noise_cols = ['InstanceNumber', 'Laterality_Clean_BILATERAL', 'Laterality_Clean_LEFT',
                      'Laterality_Clean_RIGHT', 'Laterality_Clean_Unknown'] if drop_noise else []
        
        meta_cols = ['ScanningSequence_SE', 'ScanningSequence_GR', 'ScanningSequence_IR', 'ScanningSequence_RM', 
                     'SequenceVariant_SK', 'SequenceVariant_SP', 'SequenceVariant_OSP', 'SequenceVariant_SS', 
                     'Manufacturer_Clean_Canon', 'Manufacturer_Clean_Fujifilm', 'Manufacturer_Clean_GE', 
                     'Manufacturer_Clean_Hitachi', 'Manufacturer_Clean_Philips', 'Manufacturer_Clean_Siemens',
                     'Manufacturer_Clean_Toshiba', 'Laterality_Clean_BILATERAL', 'Laterality_Clean_LEFT', 'Laterality_Clean_RIGHT', 
                     'Laterality_Clean_Unknown', 'PatientSex_Clean_Female', 'PatientSex_Clean_Male', 'PatientSex_Clean_Undefined',
                     'MRAcquisitionType_Clean_2D', 'MRAcquisitionType_Clean_3D', 'MRAcquisitionType_Clean_UNKNOWN', 'SliceNumberRate',
                     'InstanceNumber', 'Rows', 'Columns', 'MagneticFieldStrength', 'FlipAngle', 'Fluid_Sensitive', 'Fat_Suppression']   
        
        # one row per study: the 3 planes side by side
        wide = cfg.data.LIGHT_DF.pivot_table(index="StudyInstanceUID", columns="Anatomical_Plane",
                                    values=meta_cols, aggfunc="first")
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]

        def feats(df_):      # metadata rows in the SAME order as df_ (NaN if a study is missing)
            return wide.reindex(df_["StudyInstanceUID"].values).values.astype(float)

        def labels(df_):
            return (df_[cfg.glob.LABELS].values >= 0.5).astype(int)

        # GT studies may also sit inside self.train -> remove them when predicting GT
        gt_ids = set(self.GT_VAL["StudyInstanceUID"])
        train_no_gt = self.train[~self.train["StudyInstanceUID"].isin(gt_ids)]

        X_tr,  Y_tr  = feats(self.train),   labels(self.train)
        X_trg, Y_trg = feats(train_no_gt),  labels(train_no_gt)
        X_val, Y_val = feats(self.val),     labels(self.val)
        X_gt,  Y_gt  = feats(self.GT_VAL),  labels(self.GT_VAL)

        def prep(X_fit, X_other):    # fill NaN + scale, fitted on the training part only
            # pipe = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()).fit(X_fit)
            return X_fit , X_other #pipe.transform(X_fit), pipe.transform(X_other)

        Xa_tr, Xa_val = prep(X_tr, X_val)      # train -> val
        Xb_tr, Xb_gt = prep(X_trg, X_gt)       # train without GT -> GT

        factories = {
            "LR":   lambda: LogisticRegression(C=0.1, max_iter=1000),
            "LGBM": lambda: LGBMClassifier(n_estimators=100, learning_rate=0.05,
                                           random_state=42, verbose=-1),
            "HGB":  lambda: HistGradientBoostingClassifier(random_state=42, max_iter=100),
        }

        def auc_per_label(Y, P):
            out = []
            for j in range(Y.shape[1]):
                try:
                    out.append(roc_auc_score(Y[:, j], P[:, j]))
                except ValueError:             # only one class in this label
                    out.append(np.nan)
            return out

        n_lab = len(cfg.glob.LABELS)
        self.meta_preds, val_auc, gt_auc = {}, {}, {}
        self.meta_models = {name: [] for name in factories}   # 12 fitted models per method
        self.meta_cols_used = list(wide.columns)              # exact feature order
        self.meta_wide = wide
        for name, make in factories.items():
            p_val = np.zeros((len(X_val), n_lab))
            p_gt = np.zeros((len(X_gt), n_lab))
            for j in range(n_lab):
                m_val = make().fit(Xa_tr, Y_tr[:, j])          # train -> val
                m_gt  = make().fit(Xb_tr, Y_trg[:, j])         # train without GT -> GT
                p_val[:, j] = m_val.predict_proba(Xa_val)[:, 1]
                p_gt[:, j]  = m_gt.predict_proba(Xb_gt)[:, 1]
                self.meta_models[name].append(m_val)           # keep the fitted model
            self.meta_preds[name] = {"val": p_val, "gt": p_gt}
            val_auc[name] = auc_per_label(Y_val, p_val)
            gt_auc[name] = auc_per_label(Y_gt, p_gt)

        if verbose:
            print("metadata features:", len(wide.columns), "| train:", len(X_tr),
                  "| val:", len(X_val), "| GT:", len(X_gt))
            for j, lab in enumerate(cfg.glob.LABELS):
                row = " | ".join(f"{n}: val {val_auc[n][j]:.3f} GT {gt_auc[n][j]:.3f}"
                                 for n in factories)
                print(f"{lab:17s} {row}")
            print("-" * 90)
            for n in factories:
                print(f"{n:5s} macro  val AUC = {np.nanmean(val_auc[n]):.4f} | "
                      f"GT AUC = {np.nanmean(gt_auc[n]):.4f}")

        macro_val = {n: np.nanmean(val_auc[n]) for n in factories}
        self.meta_name = max(macro_val, key=macro_val.get)
        self.meta_val = self.meta_preds[self.meta_name]["val"]
        self.meta_gt = self.meta_preds[self.meta_name]["gt"]
        self.y_val_bin, self.y_gt_bin = Y_val, Y_gt
      
        print("best metadata model (by val):", self.meta_name)

      
    @torch.no_grad()
    def predict(self, ds):
        """Sigmoid probabilities (N, 12) in the same row order as ds (shuffle=False)."""
        loader = DataLoader(ds, batch_size=cfg.train.BATCH_SIZE, shuffle=False, num_workers=4,
                            pin_memory=True, collate_fn=mri_collate_fn)
        self.model.eval()
        out = []
        for batch in tqdm(loader, desc="Predict"):
            modalities, meta, _ = self._unpack(batch)
            if self.scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    logits = self.model(modalities, meta)
            else:
                logits = self.model(modalities, meta)
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.vstack(out)
 
    def blend_search(self, ckpt_path="best_model_checkpoint.pth", **kwargs):
        """Load the best image checkpoint, predict val + GT, choose the blend weight once."""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        unwrap(self.model).load_state_dict(ckpt['model'])
 
        self.img_val = self.predict(self.val_ds)        # (N_val, 12)
        self.img_gt = self.predict(self.gt_val_ds)      # (N_gt, 12)
        print("image only   | val AUC: %.4f | GT AUC: %.4f"
              % (macro_auc(self.y_val_bin, self.img_val), macro_auc(self.y_gt_bin, self.img_gt)))
        print("metadata only| val AUC: %.4f | GT AUC: %.4f  (%s)"
              % (macro_auc(self.y_val_bin, self.meta_val), macro_auc(self.y_gt_bin, self.meta_gt),
                 self.meta_name))
 
        res = search_blend_weight(self.img_val, self.meta_val, self.y_val_bin,
                                  self.img_gt, self.meta_gt, self.y_gt_bin, **kwargs)
        self.best_w = res["final_w"]
        return res
        
    def log_blend(self, epoch, weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        """Blended macro AUC for a few fixed weights, using this epoch's image predictions."""
        img_val, img_gt = self.epoch_preds["val"], self.epoch_preds["gt"]
        row = {"epoch": epoch}
        for w in weights:
            row[f"val_{w}"] = macro_auc(self.y_val_bin, blend(img_val, self.meta_val, w))
            row[f"gt_{w}"] = macro_auc(self.y_gt_bin, blend(img_gt, self.meta_gt, w))
        self.blend_log.append(row)
 
        print("blend weight w : " + "  ".join(f"{w:>6}" for w in weights))
        print("      val AUC  : " + "  ".join(f"{row[f'val_{w}']:.4f}" for w in weights))
        print("      GT  AUC  : " + "  ".join(f"{row[f'gt_{w}']:.4f}" for w in weights))
        print(f"      gain on val (best w vs w=0): "
              f"{max(row[f'val_{w}'] for w in weights) - row['val_0.0']:+.4f}")    
        
    def log_label_weights(self, epoch, **kwargs):
        """Fit per-label weights on this epoch's val predictions and print the result."""
        if not hasattr(self, "label_w_log"):
            self.label_w_log = []
        res = fit_label_weights(self.epoch_preds["val"], self.meta_val, self.y_val_bin,
                                self.epoch_preds["gt"], self.meta_gt, self.y_gt_bin, **kwargs)
        res["epoch"] = epoch
        self.label_w_log.append(res)
        print_label_weights(res, epoch)
        return res    
        
    def train_one_epoch_accumulation(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_preds, all_targets = [], []
    
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        self.optimizer.zero_grad()
    
        for i, batch in enumerate(progress_bar):
            modalities, meta, targets = self._unpack(batch)
            if i == 0:
                print(len(modalities), modalities[0].shape)
    
            if self.scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = self.model(modalities, meta)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.accumulation_step
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(modalities, meta)
                loss = self.criterion(outputs, targets)
                loss = loss / self.accumulation_step
                loss.backward()
    
            if (i + 1) % self.accumulation_step == 0 or (i + 1) == len(self.train_loader):
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
    
            running_loss += loss.item() * self.accumulation_step * targets.size(0)
    
            # detach + float, so numpy works and fp16 is handled
            all_preds.append(torch.sigmoid(outputs.detach().float()).cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
            progress_bar.set_postfix({'loss': f"{loss.item() * self.accumulation_step:.4f}"})
    
        epoch_loss = running_loss / len(self.train_loader.dataset)
    
        all_preds = np.vstack(all_preds)        # <-- missing before
        all_targets = np.vstack(all_targets)    # <-- missing before
    
        auc_scores = []
        for j in range(all_targets.shape[1]):
            y_true = all_targets[:, j]
            y_pred = all_preds[:, j]
            if not np.array_equal(y_true, y_true.astype(bool)):
                y_true = (y_true >= 0.5).astype(int)
            try:
                auc_scores.append(roc_auc_score(y_true, y_pred))
            except ValueError:
                auc_scores.append(np.nan)
    
        epoch_auc = np.nanmean(auc_scores)
        return epoch_loss, epoch_auc

    def valid_one_epoch(self, epoch , data_loader, tag=None):
        self.model.eval()
        running_loss = 0.0
        all_preds, all_targets = [], []
    
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} [Valid]")
    
        with torch.no_grad():
            for batch in progress_bar:
                modalities, meta, targets = self._unpack(batch)
    
                if self.scaler is not None:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self.model(modalities, meta)
                        loss    = self.criterion(outputs, targets)
                else:
                    outputs = self.model(modalities, meta)
                    loss    = self.criterion(outputs, targets)
    
                running_loss += loss.item() * targets.size(0)
                
                # Apply sigmoid to convert logits to probabilities [0, 1]
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(targets.cpu().numpy())
    
        # Combine batches into 2D arrays of shape (N_samples, 12)
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        epoch_loss = running_loss / len(data_loader.dataset)
    
        # Compute Macro ROC-AUC across all 12 targets safely
        num_targets = all_targets.shape[1]
        auc_scores = []
    
        for i in range(num_targets):
            y_true = all_targets[:, i]
            y_pred = all_preds[:, i]
    
            # Binarize soft labels if validation set contains continuous targets
            if not np.array_equal(y_true, y_true.astype(bool)):
                y_true = (y_true >= 0.5).astype(int)
    
            try:
                auc = roc_auc_score(y_true, y_pred)
                auc_scores.append(auc)
            except ValueError:
                # Triggered if a validation fold contains only 0s or only 1s for target i
                auc_scores.append(np.nan)
    
        # Average valid target scores matching the competition metric formula[cite: 1]
        epoch_auc = np.nanmean(auc_scores)
        if tag is not None:               
            self.epoch_preds[tag] = all_preds
        return epoch_loss, epoch_auc        

    def training(self):
        print(f"Starting Training on device: {self.device}")        
        # self.meta_trainer(drop_noise=False, verbose=True)
        def make_chunks(df, n_chunks, seed=None):
            shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
            chunks = np.array_split(shuffled, n_chunks)
            return chunks
    
        num_cycles = cfg.train.NUM_CYCLE
        n_chunks = cfg.train.NUM_CHUNKS

        resume_chunk = self.resume_epoch - (self.resume_cycle * n_chunks) + 1
        if resume_chunk >= n_chunks:
            self.resume_cycle += 1
            resume_chunk = 0
        resume_chunk = max(0, resume_chunk)
        
        for cycle in range(self.resume_cycle, num_cycles):   # start from saved cycle, not 0
            self.current_cycle = cycle
            chunks = make_chunks(self.train, n_chunks=n_chunks, seed=cycle)
            stop_training = False
            self._unfreeze_slice_encoder(cycle)
            
            self.val_ds = KneeDataset(train_labeled=self.val, series_merged=cfg.data.LIGHT_DF, split='train', augmentation=False)
            self.valid_loader = DataLoader(
                dataset=self.val_ds, batch_size=cfg.train.BATCH_SIZE, shuffle=False,
                num_workers=2, pin_memory=True, collate_fn=mri_collate_fn
            )
            self.gt_val_ds = KneeDataset(train_labeled=self.GT_VAL, series_merged=cfg.data.LIGHT_DF, split='train', augmentation=False)
            self.gt_valid_loader = DataLoader(
                dataset=self.gt_val_ds, batch_size=cfg.train.BATCH_SIZE, shuffle=False,
                num_workers=2, pin_memory=True, collate_fn=mri_collate_fn
            )
            
            for i in range(resume_chunk , n_chunks):
                epoch = cycle * n_chunks + i
                
                self.current_epoch = epoch
                chunk_df = chunks[i].reset_index(drop=True)
                train_ds = KneeDataset(train_labeled=chunk_df, series_merged=cfg.data.LIGHT_DF,
                                       split='train', augmentation=True)
                self.train_loader = DataLoader(
                    dataset=train_ds, batch_size=cfg.train.BATCH_SIZE, shuffle=True,
                    num_workers=4, pin_memory=True, collate_fn=mri_collate_fn
                )
    
                print(f"Epoch {epoch} — cycle {cycle}, chunk {i}, size {len(chunk_df)},  image_size = {cfg.glob.IMAGE_SIZE} | batch = {cfg.train.BATCH_SIZE} | accumulation ={cfg.train.ACCUMULATION_STEP}|LR:{self.optimizer.param_groups[0]['lr']:.6f}")
                tr_loss , tr_auc= self.train_one_epoch_accumulation(epoch)
                val_loss, val_auc = self.valid_one_epoch(epoch , self.valid_loader, tag="val")
                gt_val_loss, gt_val_auc = self.valid_one_epoch(epoch , self.gt_valid_loader, tag="gt")
                self.log_blend(epoch)
                self.log_label_weights(epoch)
                self.train_loss.append(tr_loss)
                self.valid_loss.append(val_loss)
                self.valid_auc.append(val_auc)
                
                print(f"\nEpoch {epoch} Summary: Train Loss = {tr_loss:.4f} | Train AUC = {tr_auc:.4f} | "
                      f"Valid Loss = {val_loss:.4f} | Valid AUC = {val_auc:.4f} | "
                      f"GT Valid Loss = {gt_val_loss:.4f} | GT Valid AUC = {gt_val_auc:.4f} | "
                      f"LR0:{self.optimizer.param_groups[0]['lr']:.6f} |  LR1:{self.optimizer.param_groups[1]['lr']:.6f}\n")
                
                if epoch > cfg.train.WARMUP_EPOCHS and (val_auc - self.best_auc) < cfg.train.MIN_DELTA:
                    self.stall += 1
                else:
                    self.stall = 0
                    
                if epoch <= cfg.train.WARMUP_EPOCHS:
                    if val_auc > self.best_auc:
                        print(f"Validation AUC improved from {self.best_auc:.4f} to {val_auc:.4f}. Saving best_model.pth...\n")
                        self.best_auc = val_auc
                        self.patience = 0
                        self._save_checkpoint( f"{cfg.data.save_chkp}/best_model_checkpoint.pth")
                        
                else:
                    if val_auc > self.best_auc:
                        print(f"Validation AUC improved from {self.best_auc:.4f} to {val_auc:.4f}. Saving best_model.pth...\n")
                        self.best_auc = val_auc
                        self.patience = 0
                        self._save_checkpoint(f"{cfg.data.save_chkp}/best_model_checkpoint.pth")
                        
                    else:
                        self.patience += 1
                        
                self._save_checkpoint( f"{cfg.data.save_chkp}/last_checkpoint.pth")
                
                if self.patience >= cfg.train.PATIENCE or self.stall >= cfg.train.PATIENCE:
                    stop_training = True
                    break

                del self.train_loader, train_ds, chunk_df
                cleanup(f"epoch {epoch}")
                    
            self.scheduler.step() 
            resume_chunk = 0
            if hasattr(self, 'valid_loader'):
                del self.valid_loader
            if hasattr(self, 'gt_valid_loader'):
                del self.gt_valid_loader
            if hasattr(self, 'train_loader'):
                del self.train_loader
            gc.collect()
            cleanup(f"cycle {cycle}")
            if stop_training:
                break
    
        plot_loss(self.valid_loss, self.train_loss, self.valid_auc, 'loss_trend.png')
        self.blend_search()
        res = fit_label_weights(self.img_val, self.meta_val, self.y_val_bin,
                            self.img_gt,  self.meta_gt,  self.y_gt_bin)
        print_label_weights(res, epoch)
        
        print(f"Training Complete. Best Validation AUC: {self.best_auc:.4f}")  

