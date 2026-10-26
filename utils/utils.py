
import threading, time, psutil, os, shutil
import warnings
warnings.filterwarnings("ignore")
import os
import random
import time
import numpy as np
import gc

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from sklearn.model_selection import KFold

import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from IPython.display import clear_output
clear_output()
            
from configs import get_config
cfg = get_config()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Warning: the following lines can slow down training performance slightly
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ──────────────────────────────────────────────────────────────────────────────
# Plotting helper
# ──────────────────────────────────────────────────────────────────────────────
def plot_loss(val_losses, train_losses, val_auc, path: str) -> None:
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label="Train Loss",  color="blue",  marker="o", linewidth=2)
    plt.plot(epochs, val_losses,   label="Val Loss",    color="red",   marker="o", linewidth=2)
    plt.plot(epochs, val_auc,      label="Val AUC",     color="green", marker="o", linewidth=2)
    # plt.title(f"Fold {fold} – Loss & AUC over Epochs", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {os.path.abspath(path)}")

# ── HELPER FUNCTION ───────────────
# ── HELPER FUNCTION ───────────────
import psutil, os, ctypes

def free_ram():
    gc.collect()
    torch.cuda.empty_cache()
    ctypes.CDLL("libc.so.6").malloc_trim(0)   # gives freed memory back to the OS

def _cgroup_mem():
    for cur, lim in [("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
                     ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "/sys/fs/cgroup/memory/memory.limit_in_bytes")]:
        try:
            return int(open(cur).read()), open(lim).read().strip()
        except Exception:
            pass
    return None, None

def start_mem_watcher(path="/kaggle/working/mem_log.txt", every=30):
    def loop():
        main = psutil.Process(os.getpid())
        while True:
            try:
                kids = main.children(recursive=True)
                kids_rss = sum(k.memory_info().rss for k in kids)
                cur, lim = _cgroup_mem()
                shm = shutil.disk_usage("/dev/shm").used
                gpu = " ".join(f"gpu{i}={torch.cuda.memory_reserved(i)/1e9:.1f}GB"
                               for i in range(torch.cuda.device_count()))
                with open(path, "a") as f:
                    f.write(f"{time.strftime('%H:%M:%S')} main={main.memory_info().rss/1e9:.1f}GB "
                            f"workers({len(kids)})={kids_rss/1e9:.1f}GB shm={shm/1e9:.1f}GB "
                            f"cgroup={(cur or 0)/1e9:.1f}GB limit={lim} {gpu}\n")
            except Exception:
                pass
            time.sleep(every)
    threading.Thread(target=loop, daemon=True).start()

def blend(img_p, meta_p, w, eps=1e-6):
    """final = image logit + w * metadata logit   (w = 0 -> image only)."""
    z = logit(np.clip(img_p, eps, 1 - eps)) + w * logit(np.clip(meta_p, eps, 1 - eps))
    return expit(z)
 
def macro_auc(y, p):
    a = []
    for j in range(y.shape[1]):
        try:
            # Check if column has both classes before calculating
            if len(np.unique(y[:, j])) < 2:
                continue
            score = roc_auc_score(y[:, j], p[:, j])
            if not np.isnan(score):
                a.append(score)
        except ValueError:
            pass
            
    # Return 0.5 (random guessing) or 0.0 if no valid columns could be evaluated
    if len(a) == 0:
        return float('nan') # or 0.5 depending on your preference
    return float(np.mean(a))

def search_blend_weight(img_val, meta_val, y_val, img_gt, meta_gt, y_gt,
                        weights=None, n_boot=200, seed=0, verbose=True):
    """
    Choose ONE global weight w.
      1. AUC of the blend on val for every w (val = the big split, 441 studies).
      2. Bootstrap the val studies -> standard error (SE) + 95% range of the gain over w=0.
      3. One-SE rule: take the SMALLEST w whose val AUC is within 1 SE of the best val AUC.
      4. Use the blend only if: gain 95% range is above 0 on val AND GT AUC does not drop.
    """
    if weights is None:
        weights = np.arange(0, 1.01, 0.1)
    weights = sorted({round(float(w), 2) for w in weights} | {0.0})
 
    rng = np.random.default_rng(seed)
    N = len(y_val)
    boots = rng.integers(0, N, size=(n_boot, N))
 
    val_auc, gt_auc, boot_auc = {}, {}, {}
    for w in weights:
        p_val = blend(img_val, meta_val, w)
        val_auc[w] = macro_auc(y_val, p_val)
        gt_auc[w] = macro_auc(y_gt, blend(img_gt, meta_gt, w))
        boot_auc[w] = np.array([macro_auc(y_val[b], p_val[b]) for b in boots])
    gains = {w: boot_auc[w] - boot_auc[0.0] for w in weights}
 
    best = max(weights, key=lambda w: val_auc[w])
    se = float(boot_auc[best].std())
    chosen = min(w for w in weights if val_auc[w] >= val_auc[best] - se)
    lo, hi = np.percentile(gains[chosen], [2.5, 97.5])
    gt_ok = gt_auc[chosen] >= gt_auc[0.0]
    use_blend = (chosen > 0) and (lo > 0) and gt_ok
    final_w = chosen if use_blend else 0.0
 
    if verbose:
        print(f"{'w':>5} | {'val AUC':>8} | {'val gain vs w=0 (95% range)':>30} | {'GT AUC':>7}")
        for w in weights:
            l, h = np.percentile(gains[w], [2.5, 97.5])
            mark = "  <- chosen" if w == chosen else ("  <- best on val" if w == best else "")
            print(f"{w:>5} | {val_auc[w]:8.4f} | {gains[w].mean():+8.4f} ({l:+.4f} to {h:+.4f}) | "
                  f"{gt_auc[w]:7.4f}{mark}")
        print(f"\nbest w on val = {best} | bootstrap SE = {se:.4f} | chosen w (1-SE rule) = {chosen}")
        print(f"gain at chosen w: 95% range {lo:+.4f} to {hi:+.4f} | "
              f"GT AUC {gt_auc[0.0]:.4f} -> {gt_auc[chosen]:.4f}")
        print("DECISION:", f"use blend with w = {final_w}" if use_blend
              else "do NOT blend (use the image model alone, w = 0)")
    return dict(final_w=final_w, chosen_w=chosen, best_w=best, se=se, use_blend=use_blend,
                val_auc=val_auc, gt_auc=gt_auc)
 
def unwrap(m):
    """DataParallel stores the real model under `.module` — use this
    whenever you need to reach a custom attribute (e.g. .backbone)."""
    return m.module if isinstance(m, nn.DataParallel) else m

def _z(p, eps=1e-6):
    return logit(np.clip(p, eps, 1 - eps))
 
 
def _auc1(y, s):
    try:
        return roc_auc_score(y, s)
    except ValueError:                           # only one class
        return np.nan
 


def mri_collate_fn_submit(batch):
    # batch items are: ((list_3d, meta), study_uid)
    list_3ds = [item[0][0] for item in batch]
    metas = [item[0][1] for item in batch]
    uids_out = [item[1] for item in batch]
    
    num_modalities = len(list_3ds[0])
    
    modalities_out = [
        torch.stack([list_3d[m] for list_3d in list_3ds], dim=0)
        for m in range(num_modalities)
    ]
    meta_out = torch.stack(metas, dim=0)
    
    return modalities_out, meta_out, uids_out

def safe_collate(batch):
    good = [b for b in batch if b[0] is not None]
    if not good:
        return None
    return mri_collate_fn_submit(good)

def _fit_stack(z_img, z_meta, y, C=1.0, nonneg=True):
    """Logistic regression  P(y=1) = sigmoid(a*z_img + b*z_meta + c)  with an L2 penalty on a, b.
    Starts from a=1, b=0 (= image only). Returns array [a, b, c]."""
    if y.min() == y.max():                       # one class only: nothing to learn
        return np.array([1.0, 0.0, 0.0])
    X = np.column_stack([z_img, z_meta])
    n = len(y)
    lam = 1.0 / (C * n)
 
    def f(theta):
        s = X @ theta[:2] + theta[2]
        p = expit(s)
        loss = np.mean(np.logaddexp(0.0, s) - y * s) + 0.5 * lam * (theta[0] ** 2 + theta[1] ** 2)
        r = p - y
        grad = np.array([np.mean(r * X[:, 0]) + lam * theta[0],
                         np.mean(r * X[:, 1]) + lam * theta[1],
                         np.mean(r)])
        return loss, grad
 
    bounds = [(0, None), (0, None), (None, None)] if nonneg else [(None, None)] * 3
    res = minimize(f, x0=np.array([1.0, 0.0, 0.0]), jac=True, method="L-BFGS-B", bounds=bounds)
    return res.x
 
 
def fit_label_weights(img_val, meta_val, y_val, img_gt, meta_gt, y_gt,
                      C=1.0, nonneg=True, n_splits=5, seed=0):
    """
    One (a, b, c) per label, fitted on VAL only.
      val AUC with metadata = honest: weights are fitted on 4/5 of val and scored on the other 1/5.
      GT  AUC with metadata = honest: GT is never used for fitting (weights come from all of val).
    All inputs are (N, 12): predicted probabilities (img_*, meta_*) and 0/1 labels (y_*).
    """
    L = y_val.shape[1]
    zi_v, zm_v, zi_g, zm_g = _z(img_val), _z(meta_val), _z(img_gt), _z(meta_gt)
 
    coefs = np.zeros((L, 3))
    oof = np.zeros_like(zi_v)                                   # held-out val scores
    kf = KFold(n_splits, shuffle=True, random_state=seed)
    for j in range(L):
        y = y_val[:, j]
        coefs[j] = _fit_stack(zi_v[:, j], zm_v[:, j], y, C, nonneg)       # final weights
        for tr, te in kf.split(zi_v):
            c = _fit_stack(zi_v[tr, j], zm_v[tr, j], y[tr], C, nonneg)
            oof[te, j] = expit(c[0] * zi_v[te, j] + c[1] * zm_v[te, j] + c[2])
 
    a, b = coefs[:, 0], coefs[:, 1]
    w = np.where(a > 1e-6, b / np.maximum(a, 1e-6), np.inf)
    out = dict(coefs=coefs, w=w,
               val_img=np.array([_auc1(y_val[:, j], zi_v[:, j]) for j in range(L)]),
               val_blend=np.array([_auc1(y_val[:, j], oof[:, j]) for j in range(L)]),
               gt_img=np.array([_auc1(y_gt[:, j], zi_g[:, j]) for j in range(L)]),
               gt_blend=np.array([_auc1(y_gt[:, j], a[j] * zi_g[:, j] + b[j] * zm_g[:, j])
                                  for j in range(L)]))
    return out
 
 
def print_label_weights(res, epoch=None, names=cfg.glob.LABELS):
    t = f"Epoch {epoch} | " if epoch is not None else ""
    print(f"\n{t}per-label weights  (img = image only, +meta = with metadata; val is held-out)")
    print(f"{'label':17s} | {'w=b/a':>6} | {'val img':>7} {'+meta':>7} | {'GT img':>7} {'+meta':>7}")
    for j, n in enumerate(names):
        w = "meta" if np.isinf(res["w"][j]) else f"{res['w'][j]:.2f}"
        print(f"{n:17s} | {w:>6} | {res['val_img'][j]:7.3f} {res['val_blend'][j]:7.3f} | "
              f"{res['gt_img'][j]:7.3f} {res['gt_blend'][j]:7.3f}")
    m = {k: np.nanmean(res[k]) for k in ["val_img", "val_blend", "gt_img", "gt_blend"]}
    print(f"{'MACRO':17s} | {'':>6} | {m['val_img']:7.3f} {m['val_blend']:7.3f} | "
          f"{m['gt_img']:7.3f} {m['gt_blend']:7.3f}")
    print(f"gain with metadata: val {m['val_blend'] - m['val_img']:+.4f} | GT {m['gt_blend'] - m['gt_img']:+.4f}")
 
 
def apply_label_weights(img_p, meta_p, coefs):
    """Final probabilities (N, 12) with the fitted per-label weights (use for the test set)."""
    return expit(coefs[:, 0] * _z(img_p) + coefs[:, 1] * _z(meta_p) + coefs[:, 2])

# ── collate for TRAIN, VAL  ──────────────────────────────────────

def mri_collate_fn(batch):
    num_modalities = len(batch[0][0])
    modalities_out = [
        torch.stack([item[0][m] for item in batch], dim=0)
        for m in range(num_modalities)
    ]
    meta_out = torch.stack([item[1] for item in batch])
    labels_out = torch.stack([item[2] for item in batch])
    return modalities_out, meta_out, labels_out

# ── collate for test (no labels) ──────────────────────────────────────
def mri_collate_fn_test(batch):
    num_modalities = len(batch[0][0])                      # item[0] is the list of 3 planes
    modalities_out = [
        torch.stack([item[0][m] for item in batch], dim=0)  # item[0][m], not item[m]
        for m in range(num_modalities)
    ]
    meta_out = torch.stack([item[1] for item in batch])
    return modalities_out, meta_out

import gc, os, ctypes, psutil

_libc = ctypes.CDLL("libc.so.6")
_proc = psutil.Process(os.getpid())

def cleanup(tag=""):
    gc.collect()                  # 1st: free Python objects
    torch.cuda.empty_cache()      # 2nd: give unused GPU cache back
    try:
        _libc.malloc_trim(0)      # 3rd: give free CPU memory back to the system
    except Exception:
        pass
    gpu = ", ".join(f"{torch.cuda.memory_reserved(i)/1e9:.1f}"
                    for i in range(torch.cuda.device_count()))
    print(f"[cleanup {tag}] RAM {_proc.memory_info().rss/1e9:.2f} GB | GPU reserved {gpu} GB")