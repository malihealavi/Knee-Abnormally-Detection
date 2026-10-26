import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import ast

from dataset.augmentation import (
    get_base_transform,
    get_rand_transform
)
from dataset.stack_neighbor import stack_neighbors
from dataset.loader_3d import load_dicom_images_3d 
from configs import get_config
cfg = get_config()

class KneeDataset(Dataset):
    PLANES = ['Axial', 'Coronal', 'Sagittal']
    TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
               'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", "Contusion", 'Fracture']

    def __init__(self, train_labeled: pd.DataFrame = cfg.data.TRAIN_LABELD, series_merged: pd.DataFrame = cfg.data.LIGHT_DF,
                 split: str = 'train', augmentation=True, stack_neighbor=True, meta_ref=None):
        """meta_ref = (meta_cols, mean, std) from the training dataset (use train_ds.meta_ref at test time)."""
        self.aug = augmentation
        self.split = split
        self.stack_neighbor = stack_neighbor
        self.train_labeled_df = train_labeled.to_dict(orient='records')
        self.series_merged_df = series_merged.set_index('SeriesInstanceUID')
        self.groupby_studyUID = series_merged.groupby('StudyInstanceUID').agg({
            'Anatomical_Plane': list,
            'SeriesInstanceUID': list
        })

        exclude_cols = ['StudyInstanceUID', 'SeriesInstanceUID', 'file_path', 'Anatomical_Plane', 'Report']
        if meta_ref is not None:
            self.meta_cols = list(meta_ref[0])          # same columns, same order as training
        else:
            self.meta_cols = [c for c in self.series_merged_df.columns
                              if c not in exclude_cols + self.TARGETS]

       
        self.num_meta = len(self.meta_cols)

    def __len__(self):
        return len(self.train_labeled_df)

    def _zeros_planes(self):
        d = {}
        for p in self.PLANES:
            t = torch.zeros((cfg.glob.NUM_SLICES, cfg.glob.IMAGE_SIZE, cfg.glob.IMAGE_SIZE), dtype=torch.float32)
            d[p] = stack_neighbors(t, window=1) if self.stack_neighbor else t
        return d

    def __getitem__(self, idx):
        row = self.train_labeled_df[idx]
        plane_order = self.PLANES
        meta_dict = {p: np.zeros(self.num_meta, dtype=np.float32) for p in plane_order}
        dict_3d = {}

        try:
            gp_row = self.groupby_studyUID.loc[row['StudyInstanceUID']]

            for i, mri_type in enumerate(gp_row['Anatomical_Plane']):
                if mri_type not in plane_order:          # NaN / unknown plane name
                    continue
                try:                                      # one bad plane must not kill the others
                    uid = gp_row['SeriesInstanceUID'][i]
                    files = ast.literal_eval(self.series_merged_df.loc[uid]['file_path'])

                    if len(files) == 0:
                        dict_3d[mri_type] = torch.zeros((cfg.glob.NUM_SLICES, cfg.glob.IMAGE_SIZE, cfg.glob.IMAGE_SIZE), dtype=torch.float32)
                        continue

                    vol = load_dicom_images_3d(files, num_imgs=cfg.glob.NUM_SLICES, img_size=cfg.glob.IMAGE_SIZE)
                    base = get_base_transform(num_slices=cfg.glob.NUM_SLICES, image_size=cfg.glob.IMAGE_SIZE, mri_keys=[mri_type])
                    dict_3d[mri_type] = base({mri_type: vol})[mri_type]
            
                except Exception as e:
                    print(f"[SKIP plane] {row['StudyInstanceUID']} {mri_type}: {e}")
                    dict_3d[mri_type] = torch.zeros((cfg.glob.NUM_SLICES, cfg.glob.IMAGE_SIZE, cfg.glob.IMAGE_SIZE), dtype=torch.float32)

            # augmentation (train only)
            if self.aug and self.split == 'train' and len(dict_3d) > 0:
                rand_aug = get_rand_transform(mri_keys=list(dict_3d.keys()))
                dict_3d = rand_aug(dict_3d)

            for key in list(dict_3d.keys()):
                val = dict_3d[key]
                val = val.as_tensor() if hasattr(val, 'as_tensor') else val
                val = val.squeeze(0) if val.ndim == 4 else val
                dict_3d[key] = val

            for p in plane_order:                         # missing plane -> zeros
                if p not in dict_3d:
                    dict_3d[p] = torch.zeros((cfg.glob.NUM_SLICES, cfg.glob.IMAGE_SIZE, cfg.glob.IMAGE_SIZE), dtype=torch.float32)

            if self.stack_neighbor:
                for key in plane_order:
                    dict_3d[key] = stack_neighbors(dict_3d[key], window=1)

        except Exception as e:
            print(f"[SKIP] Study {row.get('StudyInstanceUID')} failed: {e}")
            dict_3d = self._zeros_planes()

        list_3d = [dict_3d[p] for p in plane_order]
        meta = torch.from_numpy(np.stack([meta_dict[p] for p in plane_order]))

        if self.split == 'train':
            target = []
            for c in self.TARGETS:
                try:
                    target.append(float(row[c]))
                except Exception:
                    target.append(0.0)
            return list_3d, meta, torch.tensor(target, dtype=torch.float32)
        return list_3d, meta