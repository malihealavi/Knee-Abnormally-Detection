"""
In-domain SSL pretraining for the DINOv2 slice encoder — Knee MRI
===================================================================

Why this method (short version):
- The SliceEncoder backbone is DINOv2, which was itself pretrained with
  "self-distillation" (a student network learns to match a teacher
  network's output; the teacher is a slow-moving average of the student —
  no labels needed at all). Continuing pretraining with the SAME technique
  on your own knee MRI slices is the most natural way to shift the model's
  features from "natural photos" to "knee MRI" before you ever touch labels.
- Alternatives like MAE (mask patches, reconstruct pixels) or SimCLR/MoCo
  (contrastive, needs big batches or a memory bank) either need much more
  data/epochs to pay off, or fit a ViT-DINOv2 starting point less naturally.
- This script uses a LIGHTER version of the full DINOv2 recipe (it skips
  the extra patch-level loss and regularizer DINOv2 adds on top of DINOv1)
  since those mainly help at ImageNet scale and cost a lot of extra
  compute for a Kaggle GPU budget, without much payoff at this data size.

What it does:
1. Builds a slice-level dataset from every knee MRI series (all planes /
   sequence types pooled together — no labels used or needed; the same
   SliceEncoder will see all of them later, so SSL should too).
2. For each slice, builds a 3-channel image from the slice + its 2 spatial
   neighbors (matches how your SliceEncoder later sees data).
3. Runs "multi-crop" self-distillation: a student sees 2 large crops + a
   few small crops of the same slice, a teacher (EMA = running average of
   the student) only sees the large crops, and the student is trained to
   predict the teacher's output for every crop it doesn't share a view with.
4. Saves the teacher's backbone in HuggingFace format so you can load it
   exactly the way you already do:
       AutoModel.from_pretrained(SSL_OUTPUT_DIR, local_files_only=True)

Tuning knobs are all in the CONFIG block below — see the notes at the
bottom of the chat message for what to change and why.

"""
import os
import glob
import random
import warnings
import numpy as np
import torch
import torch.nn as nn

import math
import pickle
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader, RandomSampler

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
import pickle
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader, RandomSampler

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from transformers import AutoModel

# [FIX 3] needed only for the linear probe — sklearn ships by default on Kaggle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning

# ─────────────────────────────────────────────
# Config — tune these for your GPU / time budget
# ─────────────────────────────────────────────
SERIES_ROOT    = "/kaggle/input/competitions/rsna-knee-abnormality-detection/train_series"
BACKBONE_PATH  = "/kaggle/input/models/metaresearch/dinov2/pytorch/small/1"  # your existing DINOv2-small checkpoint
SSL_OUTPUT_DIR = "/kaggle/working/dinov2_knee_ssl"  # matches the path your SliceEncoder already expects

IMAGE_SIZE_GLOBAL = 256   # matches your main pipeline's IMAGE_SIZE
IMAGE_SIZE_LOCAL  = 96
NUM_GLOBAL_CROPS  = 2
NUM_LOCAL_CROPS   = 4      # raise to 6-8 if you have GPU memory to spare

BATCH_SIZE      = 32
ACCUM_STEPS     = 2        # effective batch size = BATCH_SIZE * ACCUM_STEPS
EPOCHS          = 20
STEPS_PER_EPOCH = 500      # caps how long one epoch takes; raise if you have time budget
WARMUP_STEPS    = 2 * STEPS_PER_EPOCH

# --- [FIX 2] two learning rates instead of one ---------------------------
HEAD_LR      = 1e-3 * (BATCH_SIZE * ACCUM_STEPS) / 256   # same formula/value as your original BASE_LR — used for the SSL head only
BACKBONE_LR  = 1e-5                                        # NEW, much lower — only the unfrozen last blocks use this
BASE_LR      = HEAD_LR   # kept so anything else referencing BASE_LR still works, unchanged meaning for the head
# ---------------------------------------------------------------------------

WEIGHT_DECAY = 0.04
OUT_DIM      = 8192        # number of "prototypes" for the self-distillation loss
                            # (smaller than the usual 65536 since our dataset is far
                            # smaller than ImageNet)
STUDENT_TEMP = 0.1
TEACHER_TEMP_WARMUP = 0.04
TEACHER_TEMP = 0.07
TEACHER_TEMP_WARMUP_EPOCHS = 5
TEACHER_MOMENTUM_BASE = 0.996
CENTER_MOMENTUM = 0.9

SAVE_EVERY  = 4
NUM_WORKERS = 4
SEED        = 42

# --- [FIX 1] how much of the backbone stays trainable ---------------------
NUM_TRAINABLE_BLOCKS = 2     # only the LAST 2 transformer blocks train; set to 1 for an even more conservative fine-tune
UNFREEZE_FINAL_NORM  = True  # also train the small final layernorm right after the blocks (cheap, sits right on the CLS output)
# ---------------------------------------------------------------------------

# --- [FIX 3] linear probe settings -----------------------------------------
RUN_LINEAR_PROBE   = True   # keep False until load_verified_probe_studies() below is filled in — script runs unchanged if False
PROBE_EVERY        = 4      # epochs between probe checks
PROBE_N_SLICES     = 5       # slices sampled per study for feature extraction (evenly spaced)
PROBE_KFOLDS       = 5
SSL_OUTPUT_DIR_BEST = SSL_OUTPUT_DIR + "_best_probe"  # separate folder — does NOT touch your existing SSL_OUTPUT_DIR checkpoints

LABEL_COLUMNS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
               'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", "Contusion", 'Fracture']
# ---------------------------------------------------------------------------

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────────
# 1. Build the list of usable series (reuse SERIES_FILES if you already built it)
# ─────────────────────────────────────────────
def build_series_index(root_dir):
    series_dirs = glob.glob(os.path.join(root_dir, "*", "*"))
    series_files = []
    for sdir in series_dirs:
        if not os.path.isdir(sdir):
            continue
        files = glob.glob(os.path.join(sdir, "*.dcm"))
        if len(files) < 3:
            continue
        files.sort(key=lambda f: int(pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber))
        series_files.append(files)
    return series_files

with open('/kaggle/input/datasets/malihealavi/knee-abnormally-dataset/series_files.pkl', "rb") as f:
    SERIES_FILES = pickle.load(f)
print(len(SERIES_FILES))

try:
    ALL_SERIES_FILES = SERIES_FILES  # reuse if this notebook already built it
    print(f"Reusing existing SERIES_FILES ({len(ALL_SERIES_FILES)} series)")
except NameError:
    ALL_SERIES_FILES = build_series_index(SERIES_ROOT)
    print(f"Indexed {len(ALL_SERIES_FILES)} series")


# ─────────────────────────────────────────────
# 2. Slice-level dataset — every slice of every series, no labels
# ─────────────────────────────────────────────
def load_slice_as_array(path):
    dicom = pydicom.dcmread(path)
    try:
        arr = apply_voi_lut(dicom.pixel_array, dicom)
    except Exception:
        arr = dicom.pixel_array
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return arr


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class DINOMultiCrop:
    """Student sees NUM_GLOBAL_CROPS large crops + NUM_LOCAL_CROPS small crops.
    Teacher only ever sees the large crops (handled in the training loop)."""

    def __init__(self):
        flip_and_jitter = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            # brightness/contrast only — no hue/saturation, since the 3 channels
            # here are neighboring slices, not real RGB color
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4)], p=0.6),
            T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.4),
        ])
        normalize = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

        self.global_tf = [T.Compose([
            T.RandomResizedCrop(IMAGE_SIZE_GLOBAL, scale=(0.4, 1.0),
                                 interpolation=T.InterpolationMode.BICUBIC),
            flip_and_jitter, normalize,
        ]) for _ in range(NUM_GLOBAL_CROPS)]

        self.local_tf = T.Compose([
            T.RandomResizedCrop(IMAGE_SIZE_LOCAL, scale=(0.05, 0.4),
                                 interpolation=T.InterpolationMode.BICUBIC),
            flip_and_jitter, normalize,
        ])

    def __call__(self, pil_img):
        crops = [tf(pil_img) for tf in self.global_tf]
        crops += [self.local_tf(pil_img) for _ in range(NUM_LOCAL_CROPS)]
        return crops


class SliceSSLDataset(Dataset):
    """One item = one 2.5D slice (itself + its 2 neighbors as 3 channels),
    pooled across ALL series/planes/sequence types."""

    def __init__(self, series_files, transform):
        self.series_files = series_files
        self.transform = transform
        self.index = [(si, zi) for si, files in enumerate(series_files) for zi in range(len(files))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        si, zi = self.index[idx]
        files = self.series_files[si]
        n = len(files)
        neighbor_idxs = [max(0, min(n - 1, zi + o)) for o in (-1, 0, 1)]
        channels = [load_slice_as_array(files[i]) for i in neighbor_idxs]
        img = np.stack(channels, axis=-1)  # (H, W, 3), float in [0,1]
        pil_img = Image.fromarray((img * 255).astype(np.uint8), mode="RGB")
        return self.transform(pil_img)


def dino_collate(batch):
    n_crops = len(batch[0])
    return [torch.stack([sample[c] for sample in batch]) for c in range(n_crops)]


dataset = SliceSSLDataset(ALL_SERIES_FILES, DINOMultiCrop())
sampler = RandomSampler(dataset, replacement=True, num_samples=STEPS_PER_EPOCH * BATCH_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                     collate_fn=dino_collate, num_workers=NUM_WORKERS,
                     pin_memory=True, drop_last=True)


# ─────────────────────────────────────────────
# 3. Student / teacher model
# ─────────────────────────────────────────────
class HFBackboneWrapper(nn.Module):
    """Wraps the HF DINOv2 model to return just the CLS embedding —
    identical to what your SliceEncoder does."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x).last_hidden_state[:, 0, :]


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=OUT_DIM, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class DINOModel(nn.Module):
    def __init__(self, backbone, embed_dim):
        super().__init__()
        self.backbone = HFBackboneWrapper(backbone)
        self.head = DINOHead(embed_dim)

    def forward(self, x):
        return self.head(self.backbone(x))


# --- [FIX 1] freeze most of the backbone -----------------------------------
def freeze_backbone_except_last_n(backbone, n_trainable_blocks=NUM_TRAINABLE_BLOCKS,
                                   unfreeze_final_norm=UNFREEZE_FINAL_NORM):
    """
    Freezes all backbone parameters (requires_grad=False, so no gradient
    is computed and the optimizer never touches them), then unfreezes only
    the LAST `n_trainable_blocks` transformer blocks (and, optionally, the
    final layernorm right after them).

    Expects the standard HuggingFace Dinov2Model layout:
        backbone.embeddings            <- patch/pos embeddings, stays frozen
        backbone.encoder.layer[i]      <- transformer blocks
        backbone.layernorm             <- final norm, optionally trainable

    If your checkpoint has a different internal structure, this will raise
    a clear error rather than silently freezing the wrong thing.
    """
    for p in backbone.parameters():
        p.requires_grad = False

    try:
        blocks = backbone.encoder.layer
    except AttributeError as e:
        raise AttributeError(
            "Could not find backbone.encoder.layer — this backbone's "
            "internal structure is different than expected. Run "
            "print(backbone) to inspect it and update this function to "
            "point at the actual transformer-block list before continuing."
        ) from e

    n_total = len(blocks)
    n_trainable_blocks = min(n_trainable_blocks, n_total)
    for block in blocks[n_total - n_trainable_blocks:]:
        for p in block.parameters():
            p.requires_grad = True

    if unfreeze_final_norm and hasattr(backbone, "layernorm"):
        for p in backbone.layernorm.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbone.parameters())
    print(f"[FIX 1] Backbone: {trainable:,} / {total:,} params trainable "
          f"({100 * trainable / total:.1f}%) — last {n_trainable_blocks}/{n_total} blocks"
          f"{' + final layernorm' if unfreeze_final_norm else ''} unfrozen")
# ---------------------------------------------------------------------------


student_backbone = AutoModel.from_pretrained(BACKBONE_PATH, local_files_only=True)
teacher_backbone = AutoModel.from_pretrained(BACKBONE_PATH, local_files_only=True)
embed_dim = student_backbone.config.hidden_size

# [FIX 1] apply the freeze to the student's backbone before it's wrapped.
# (Not needed on teacher_backbone — a few lines below, ALL teacher params
# are set to requires_grad=False anyway, since the teacher is never
# trained by gradient descent, only by the EMA update further down.)
freeze_backbone_except_last_n(student_backbone)

student = DINOModel(student_backbone, embed_dim).to(device)
teacher = DINOModel(teacher_backbone, embed_dim).to(device)
teacher.load_state_dict(student.state_dict())  # sync the randomly-init heads too
for p in teacher.parameters():
    p.requires_grad = False


def unwrap(m):
    """DataParallel stores the real model under `.module` — use this
    whenever you need to reach a custom attribute (e.g. .backbone)."""
    return m.module if isinstance(m, nn.DataParallel) else m


if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
    student = nn.DataParallel(student)
    teacher = nn.DataParallel(teacher)


# ─────────────────────────────────────────────
# 4. DINO loss (self-distillation cross-entropy + centering, no labels)
# ─────────────────────────────────────────────
class DINOLoss(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(TEACHER_TEMP_WARMUP, TEACHER_TEMP, TEACHER_TEMP_WARMUP_EPOCHS),
            np.full(max(EPOCHS - TEACHER_TEMP_WARMUP_EPOCHS, 0), TEACHER_TEMP),
        ))

    def forward(self, student_outputs, teacher_outputs, epoch):
        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        teacher_probs = [F.softmax((t - self.center) / temp, dim=-1).detach() for t in teacher_outputs]
        student_logp  = [F.log_softmax(s / STUDENT_TEMP, dim=-1) for s in student_outputs]

        loss, n_terms = 0.0, 0
        for ti, tp in enumerate(teacher_probs):
            for si, slp in enumerate(student_logp):
                if si == ti:  # never compare a view against itself
                    continue
                loss += torch.sum(-tp * slp, dim=-1).mean()
                n_terms += 1
        loss = loss / n_terms

        with torch.no_grad():
            batch_center = torch.cat(teacher_outputs).mean(dim=0, keepdim=True)
            self.center = self.center * CENTER_MOMENTUM + batch_center * (1 - CENTER_MOMENTUM)
        return loss


dino_loss = DINOLoss(OUT_DIM).to(device)

# --- [FIX 2] two parameter groups, two learning rates -----------------------
# Only params with requires_grad=True need gradients/updates — after
# freeze_backbone_except_last_n(), that's just the last few blocks (+ maybe
# the final layernorm) in the backbone, plus the whole head.
backbone_trainable_params = [p for p in unwrap(student).backbone.parameters() if p.requires_grad]
head_params = list(unwrap(student).head.parameters())

BACKBONE_GROUP_IDX, HEAD_GROUP_IDX = 0, 1
optimizer = torch.optim.AdamW(
    [
        {"params": backbone_trainable_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_LR},
    ],
    weight_decay=WEIGHT_DECAY,
)
# ---------------------------------------------------------------------------

scaler = torch.cuda.amp.GradScaler()


def cosine_lr(step, total_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


def cosine_momentum(epoch, total_epochs, base=TEACHER_MOMENTUM_BASE):
    return 1 - (1 - base) * (math.cos(math.pi * epoch / total_epochs) + 1) / 2


@torch.no_grad()
def update_teacher(m):
    # Still correct with a partially frozen backbone: frozen student params
    # never change, and a frozen pair (pt == ps already) stays exactly equal
    # under this EMA formula (pt*m + ps*(1-m) == ps when pt == ps). Only the
    # unfrozen blocks actually drift and get EMA'd normally.
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=1 - m)


FREEZE_LAST_LAYER_EPOCHS = 1  # standard DINO stabilization trick — without this,
                              # a randomly-initialized output layer can drag the
                              # whole network into a collapsed shortcut early on


def cancel_last_layer_grad(model, epoch):
    """Zero the output layer's gradient for the first epoch. This is what
    the official DINO code calls --freeze_last_layer (default 1 epoch)."""
    if epoch >= FREEZE_LAST_LAYER_EPOCHS:
        return
    for n, p in unwrap(model).head.named_parameters():
        if "last_layer" in n and p.grad is not None:
            p.grad = None


# ─────────────────────────────────────────────
# 4b. [FIX 3] Linear probe — the ground-truth check for feature quality
#     (loss going down does NOT mean the features are good; this does)
# ─────────────────────────────────────────────
import ast
def load_verified_probe_studies():
    global TRUE_LABELS_DF, LIGHT_DF
    final_list = []
    for _, row in TRUE_LABELS_DF.iterrows():
        study_id = row['StudyInstanceUID']
        study_df = LIGHT_DF[LIGHT_DF['StudyInstanceUID'] == study_id]
        if study_df.empty:
            print(f"  [probe] no files found for {study_id}, skipping")
            continue
        label_vector = row[LABEL_COLUMNS].values.astype(float)
        for _, series_row in study_df.iterrows():
            final_list.append((ast.literal_eval(series_row['file_path']), label_vector, study_id))
    return final_list


@torch.no_grad()
def extract_study_embedding(teacher_model, files, n_slices=PROBE_N_SLICES):
    """Mean-pools teacher CLS embeddings over a few evenly spaced 2.5D
    slices of one series — same slice-building logic as SliceSSLDataset,
    just without augmentation (resize only, no crop/flip/jitter)."""
    n = len(files)
    if n == 0:
        return None
    idxs = np.linspace(0, n - 1, num=min(n_slices, n), dtype=int)
    normalize = T.Compose([
        T.Resize((IMAGE_SIZE_GLOBAL, IMAGE_SIZE_GLOBAL), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    embeds = []
    for zi in idxs:
        zi = int(zi)
        neighbor_idxs = [max(0, min(n - 1, zi + o)) for o in (-1, 0, 1)]
        channels = [load_slice_as_array(files[i]) for i in neighbor_idxs]
        img = np.stack(channels, axis=-1)
        pil_img = Image.fromarray((img * 255).astype(np.uint8), mode="RGB")
        x = normalize(pil_img).unsqueeze(0).to(device)
        feat = teacher_model.backbone(x)  # HFBackboneWrapper -> CLS embedding
        embeds.append(feat.squeeze(0).float().cpu().numpy())
    return np.mean(embeds, axis=0)

from sklearn.model_selection import GroupKFold

def run_linear_probe(teacher_model, verified_studies, epoch):
    """
    Extracts one pooled embedding per verified study, then fits a simple
    per-label logistic regression with k-fold cross-validation (58 samples
    is too few to trust a single train/val split) and reports mean AUC
    across labels and folds. This is the real signal for whether SSL
    pretraining is helping — not the training loss.
    """
    teacher_model.eval()
    feats, labels , groups = [], [], []
    for files, label_vec , study_id in verified_studies:
        emb = extract_study_embedding(teacher_model, files)
        if emb is None:
            continue
        feats.append(emb)
        labels.append(label_vec)
        groups.append(study_id)
        
    feats = np.stack(feats)
    labels = np.array(labels)

    n_labels = labels.shape[1]
    n_groups = len(set(groups))
    n_splits = max(2, min(PROBE_KFOLDS, n_groups))
    kf = GroupKFold(n_splits=n_splits)
    per_label_aucs = []
 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        for label_idx in range(n_labels):
            y = labels[:, label_idx]
            if len(np.unique(y)) < 2:
                continue  # AUC undefined if a label has only one class overall
            fold_aucs = []
            for train_idx, val_idx in kf.split(feats, groups=groups):
                y_train = (y[train_idx] > 0.5).astype(int)
                if len(np.unique(y_train)) < 2:
                    continue
                y_val = (y[val_idx] > 0.5).astype(int)
                if len(np.unique(y_val)) < 2:
                    continue
                clf = LogisticRegression(max_iter=1000)
                clf.fit(feats[train_idx], y_train)
                probs = clf.predict_proba(feats[val_idx])[:, 1]
                fold_aucs.append(roc_auc_score(y_val, probs))
            if fold_aucs:
                per_label_aucs.append(np.mean(fold_aucs))
 
    mean_auc = float(np.mean(per_label_aucs)) if per_label_aucs else float("nan")
    print(f"  [linear probe] epoch {epoch + 1}: mean AUC over "
          f"{len(per_label_aucs)}/{n_labels} usable labels = {mean_auc:.4f}")
    return mean_auc
 



verified_probe_studies = None
if RUN_LINEAR_PROBE:
    verified_probe_studies = load_verified_probe_studies()
    print(f"[FIX 3] Loaded {len(verified_probe_studies)} verified studies "
          f"for linear-probe checks (every {PROBE_EVERY} epochs)")
class _ProbeWrapper(nn.Module):
    def __init__(self, backbone_wrapper):
        super().__init__()
        self.backbone = backbone_wrapper

baseline_container = _ProbeWrapper(
    HFBackboneWrapper(AutoModel.from_pretrained(BACKBONE_PATH, local_files_only=True))
).to(device)

baseline_auc = run_linear_probe(baseline_container, verified_probe_studies, epoch=-1)
print(f"Baseline (plain DINOv2, no SSL) probe AUC: {baseline_auc:.4f}")

best_probe_auc = -1.0
probe_history = []


# ─────────────────────────────────────────────
# 5. Training loop
# ─────────────────────────────────────────────
os.makedirs(SSL_OUTPUT_DIR, exist_ok=True)
total_steps = STEPS_PER_EPOCH * EPOCHS
global_step = 0
loss_history = []

for epoch in range(EPOCHS):
    student.train()
    teacher.eval()
    running = 0.0

    for it, crops in enumerate(loader):
        crops = [c.to(device, non_blocking=True) for c in crops]

        # --- [FIX 2] one cosine schedule, two peak LRs -----------------
        # cosine_lr(..., base_lr=1.0, ...) returns the 0..1 warmup/decay
        # multiplier by itself; multiply it by each group's own peak LR.
        mult = cosine_lr(global_step, total_steps, 1.0, WARMUP_STEPS)
        lr_backbone = BACKBONE_LR * mult
        lr_head = HEAD_LR * mult
        optimizer.param_groups[BACKBONE_GROUP_IDX]["lr"] = lr_backbone
        optimizer.param_groups[HEAD_GROUP_IDX]["lr"] = lr_head
        # ------------------------------------------------------------------

        with torch.cuda.amp.autocast():
            student_outputs = [student(c) for c in crops]
            with torch.no_grad():
                teacher_outputs = [teacher(c) for c in crops[:NUM_GLOBAL_CROPS]]
            loss = dino_loss(student_outputs, teacher_outputs, epoch) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (it + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            cancel_last_layer_grad(student, epoch)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            update_teacher(cosine_momentum(epoch, EPOCHS))

        running += loss.item() * ACCUM_STEPS
        global_step += 1

    epoch_loss = running / STEPS_PER_EPOCH
    loss_history.append(epoch_loss)
    print(f"epoch {epoch + 1:>3}/{EPOCHS}  loss={epoch_loss:.4f}  "
          f"lr_backbone={lr_backbone:.2e}  lr_head={lr_head:.2e}")

    if (epoch + 1) % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
        unwrap(teacher).backbone.backbone.save_pretrained(SSL_OUTPUT_DIR)
        print(f"  saved teacher backbone -> {SSL_OUTPUT_DIR}")

    # --- [FIX 3] periodic linear probe check ---------------------------
    if RUN_LINEAR_PROBE and ((epoch + 1) % PROBE_EVERY == 0 or epoch == EPOCHS - 1):
        probe_auc = run_linear_probe(unwrap(teacher), verified_probe_studies, epoch)
        probe_history.append((epoch + 1, probe_auc))
        if probe_auc == probe_auc and probe_auc > best_probe_auc:  # NaN-safe check
            best_probe_auc = probe_auc
            os.makedirs(SSL_OUTPUT_DIR_BEST, exist_ok=True)
            unwrap(teacher).backbone.backbone.save_pretrained(SSL_OUTPUT_DIR_BEST)
            print(f"  new best probe AUC ({best_probe_auc:.4f}) -> saved to {SSL_OUTPUT_DIR_BEST}")
    # ------------------------------------------------------------------

print("Done. Load the result the same way you already do:")
print(f'  AutoModel.from_pretrained("{SSL_OUTPUT_DIR}", local_files_only=True)')
if RUN_LINEAR_PROBE:
    print(f"Best probe AUC = {best_probe_auc:.4f}, saved separately at {SSL_OUTPUT_DIR_BEST}")
    print(f"Probe history: {probe_history}")