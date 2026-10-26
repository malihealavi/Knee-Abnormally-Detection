
from dataclasses import dataclass, field
import os
import torch
import pandas as pd

def _get_default_device() -> str:
    """Returns 'cuda' if a GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class DataPaths:
    save_chkp = './assets/checkpoint'
    TRAIN_SERIES:pd.DataFrame  = pd.read_csv('./assets/data/train_series.csv')
    TEST_SERIES :pd.DataFrame= pd.read_csv('./assets/data/test_series.csv')
    SERIES_MERGED :pd.DataFrame= pd.read_csv('./assets/data/df_merged.csv')
    TRAIN_LABELD_OLD :pd.DataFrame= pd.read_csv('./assets/data/labeld_df.csv')
    TRAIN_LABELD:pd.DataFrame = pd.read_csv('./assets/data/labeled_df_by_group.csv')
    TRAIN :pd.DataFrame = pd.read_csv('./assets/data/train.csv')
    
    TRAIN_LABELD [['Medial OA', 'Lateral OA']] = TRAIN_LABELD_OLD[['Medial OA', 'Lateral OA']]
    TRUE_LABELS_DF = TRAIN[TRAIN.notna().all(axis=1)]
    GT_VAL = TRAIN_LABELD.iloc[TRUE_LABELS_DF.index]
    TRAIN_LABELD = TRAIN_LABELD.drop(TRUE_LABELS_DF.index)
    


@dataclass
class GlobalVar:
    NUM_SLICES: int =  16
    IMAGE_SIZE :int = 256
    PLANES : list = ['Axial', 'Coronal', 'Sagittal']
    LABELS= ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 
             'PF OA', 'Effusion', 'Synovitis', "Baker's", "Contusion", 'Fracture']

@dataclass
class ModelConfig:
    d: int= 256
    nhead: int= 4
    num_layers: int= 2
    num_modalities: int= 3
    d_meta = 32


@dataclass
class TrainConfig:
    SEED: int = 42
    LR_HEAD: float = 1e-4
    LR_BB: float = 1e-5
    WEIGHT_DECAY: int = 5e-3
    BATCH_SIZE = 8
    ACCUMULATION_STEP: int  = 4
    WARMUP_EPOCHS: int  = 0
    MIN_DELTA: int  = 0.001   
    PATIENCE: int  = 6
    ENCODER_BB = 'convnext_nano'
    NUM_CYCLE: int = 30
    NUM_CHUNKS: int = 3
    TOTAL_BACKBONE_BLOCKS: int = 12   # dinov2_small has 12 transformer blocks (confirmed from your SSL run output)
    UNFREEZE_START_CYCLE : int = 2    # backbone stays fully frozen before this cycle
    CYCLES_PER_BLOCK : int     = 1    # +1 more block unfrozen every this many cycles
    MAX_UNFROZEN_BLOCKS : int  = 4    # cap — matches the "last 1-2 blocks" target from before
    TOTAL_CONV_STAGES: int = 2     



@dataclass
class InferenceConfig:
    NUM_SLICES: int =  16
    IMAGE_SIZE :int = 256
    BATCH_SIZE = 8



@dataclass
class Config:
    data: DataPaths = field(default_factory=DataPaths)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    glob : GlobalVar = field(default_factory=InferenceConfig)
    device: str = field(default_factory=_get_default_device)

def get_config() -> Config:
    return Config()