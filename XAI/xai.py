# ══════════════════════════════════════════════════════════════
#  XAI for MRI25DFusionModelViewRoutingDualEncoder
#  Needs: TARGET_COLS, SafeSubmissionDataset, safe_collate (from your submit code)
# ══════════════════════════════════════════════════════════════
import os
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from configs import get_config
cfg = get_config()


# ─────────────────────────────────────────────
# 1. Split the model forward in two parts
#    (so we can encode slices once, then hide slices cheaply)
# ─────────────────────────────────────────────
@torch.no_grad()
def encode_slices(model, mods):
    feats = []
    for s in mods:
        B, N, C, H, W = s.shape
        flat = s.reshape(B * N, C, H, W)
        fd = model.dino_slice_encoder(flat).view(B, N, -1)
        fc = model.conv_slice_encoder(flat).view(B, N, -1)
        feats.append(model.slice_fuse(torch.cat([fd, fc], dim=-1)))
    return feats                                           # list of (B, N, d)


def heads_from_feats(model, feats, masks=None):
    if masks is None:
        masks = [None] * len(feats)
    vecs    = [p(f, m) for f, p, m in zip(feats, model.slice_poolers, masks)]
    fused   = model.fusion(vecs)
    axial   = vecs[0]
    sc      = (vecs[1] + vecs[2]) / 2
    all_avg = torch.stack(vecs, dim=1).mean(dim=1)
    out = torch.cat([model.axial_head(torch.cat([axial,   fused], 1)),
                     model.sc_head(torch.cat([sc,         fused], 1)),
                     model.general_head(torch.cat([all_avg, fused], 1))], dim=1)
    return out[:, model.label_perm]                        # logits (B, 12)


# ─────────────────────────────────────────────
# 2. Deletion tests (slice level + plane level)
#    drop = logit(full) - logit(with part hidden)
#    drop > 0  → this part SUPPORTS the target
# ─────────────────────────────────────────────
@torch.no_grad()
def deletion_scores(model, mods):
    dev   = mods[0].device
    feats = encode_slices(model, mods)
    base  = heads_from_feats(model, feats)[0]              # (12,)
    P = len(feats)
    slice_drop, plane_drop = [], []
    for p in range(P):
        N = feats[p].shape[1]
        # one batch row per hidden slice
        exp   = [f.expand(N, -1, -1) for f in feats]
        masks = [None] * P
        masks[p] = torch.eye(N, dtype=torch.bool, device=dev)       # True = hidden
        out = heads_from_feats(model, exp, masks)                   # (N, 12)
        slice_drop.append((base - out).cpu().numpy())               # (N, 12)
        # hide the whole plane
        m_all = [None] * P
        m_all[p] = torch.ones(1, N, dtype=torch.bool, device=dev)
        out_p = heads_from_feats(model, feats, m_all)[0]
        plane_drop.append((base - out_p).cpu().numpy())             # (12,)
    return slice_drop, np.stack(plane_drop)                         # list[(N,12)], (P,12)


# ─────────────────────────────────────────────
# 3. Grad-CAM on both encoders (middle slice of every plane)
# ─────────────────────────────────────────────
class ActStore:
    """Grabs the last feature maps of both encoders during the forward pass."""
    def __init__(self, model):
        d = model.dino_slice_encoder.backbone
        c = model.conv_slice_encoder.backbone
        self.patch = d.config.patch_size
        self.dino, self.conv = [], []
        self.handles = [
            d.encoder.layer[-1].register_forward_pre_hook(self._dino, with_kwargs=True),
            c.stages[-1].register_forward_hook(self._conv),
        ]
    def _dino(self, mod, args, kwargs):
        self.dino.append(args[0] if args else kwargs['hidden_states'])   # (N, 1+tokens, C)
    def _conv(self, mod, inp, out):
        self.conv.append(out)                                            # (N, C, h, w)
    def close(self):
        for h in self.handles:
            h.remove()


def _up(cam, H, W):
    cam = cam[None, None].float()
    return F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)[0, 0].detach().cpu().numpy()

def _cam_conv(act, grad):                                   # classic Grad-CAM
    w = grad.mean(dim=(1, 2), keepdim=True)
    return F.relu((w * act).sum(0))

def _cam_vit(act, grad, hh, ww):                            # gradient x activation per patch
    t = (act * grad).sum(-1)[1:]                            # drop CLS token
    assert t.numel() == hh * ww, f"token grid mismatch: {t.numel()} vs {hh}x{ww}"
    return F.relu(t).view(hh, ww)


def gradcam_all_labels(model, mods, slice_idx=None):
    """Returns CAMs (12, P, H, W) for 'dino', 'conv', 'both' + middle slice images."""
    model.eval()
    mods = [m.detach().clone().float().requires_grad_(True) for m in mods]  # needed: backbones may be frozen
    H, W = mods[0].shape[-2:]
    assert all(m.shape[-2:] == (H, W) for m in mods), "planes have different H/W"
    P    = len(mods)
    mids = [(m.shape[1] // 2 if slice_idx is None else slice_idx) for m in mods]

    store = ActStore(model)
    try:
        with torch.enable_grad():
            logits = model(mods)[0]                                         # (12,)
            assert len(store.dino) == P and len(store.conv) == P
            tensors = store.dino + store.conv
            n_lab   = logits.numel()
            cam = {e: np.zeros((n_lab, P, H, W), np.float32) for e in ('dino', 'conv')}
            for k in range(n_lab):
                grads = torch.autograd.grad(logits[k], tensors,
                                            retain_graph=(k < n_lab - 1), allow_unused=True)
                for p in range(P):
                    gd, gc, m = grads[p], grads[P + p], mids[p]
                    if gd is not None:
                        cam['dino'][k, p] = _up(_cam_vit(store.dino[p][m], gd[m],
                                                         H // store.patch, W // store.patch), H, W)
                    if gc is not None:
                        cam['conv'][k, p] = _up(_cam_conv(store.conv[p][m], gc[m]), H, W)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
    finally:
        store.close()

    # scale per label across planes, so planes with no evidence stay dark
    for e in ('dino', 'conv'):
        cam[e] /= cam[e].max(axis=(1, 2, 3), keepdims=True) + 1e-8
    cam['both'] = 0.5 * (cam['dino'] + cam['conv'])
    cam['both'] /= cam['both'].max(axis=(1, 2, 3), keepdims=True) + 1e-8

    imgs = np.stack([mods[p][0, mids[p], 1].detach().cpu().numpy() for p in range(P)])  # centre channel
    return dict(cam=cam, imgs=imgs, prob=probs, mids=mids)


# ─────────────────────────────────────────────
# 4. Plots
# ─────────────────────────────────────────────
def plot_cam_grid(res, p, title, path, y_true=None, which='both'):
    img = res['imgs'][p]
    lo, hi = np.percentile(img, (1, 99))
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for k, ax in enumerate(axes.ravel()):
        ax.imshow(img, cmap='gray')
        ax.imshow(res['cam'][which][k, p], cmap='jet', alpha=0.45, vmin=0, vmax=1)
        t = f"{cfg.glob.LABELS[k]}  p={res['prob'][k]:.2f}"
        if y_true is not None:
            t += f"  GT={y_true[k]:.2f}"
        ax.set_title(t, fontsize=11); ax.axis('off')
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches='tight'); plt.close(fig)


def _heat(ax, M, xlabels, title, annotate=False, ylabels=True):
    v = np.abs(M).max() + 1e-8
    ax.imshow(M, cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto')
    ax.set_title(title); ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_yticks(range(len(cfg.glob.LABELS)))
    ax.set_yticklabels(cfg.glob.LABELS if ylabels else [''] * len(cfg.glob.LABELS), fontsize=9)
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha='center', va='center', fontsize=8)


def plot_importance(plane_drop, slice_drop, title, path):
    P = plane_drop.shape[0]
    fig, axes = plt.subplots(1, 1 + P, figsize=(5 + 5 * P, 6), gridspec_kw=dict(width_ratios=[1] + [2] * P))
    _heat(axes[0], plane_drop.T, cfg.glob.PLANES, 'Plane deletion\n(logit drop)', annotate=True)
    for p in range(P):
        N = slice_drop[p].shape[0]
        _heat(axes[1 + p], slice_drop[p].T, list(range(N)), f'{cfg.glob.PLANES[p]}: slice deletion', ylabels=False)
        axes[1 + p].set_xlabel('slice index')
    fig.suptitle(title + '   (red = supports the target, blue = works against it)', fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches='tight'); plt.close(fig)


# ─────────────────────────────────────────────
# 5. One study → all explanations
# ─────────────────────────────────────────────
def explain_study(model, mods, uid, y_true=None, out_dir='xai_out', save_figs=True, check=False):
    dev  = next(model.parameters()).device
    mods = [m.to(dev, dtype=torch.float32) for m in mods]
    model.eval()

    if check:   # my split-forward must give the same output as the real model
        with torch.no_grad():
            a = model(mods); b = heads_from_feats(model, encode_slices(model, mods))
        assert torch.allclose(a, b, atol=1e-4), f"split forward differs: {(a-b).abs().max().item()}"
        print("sanity check OK: split forward == model forward")

    with torch.no_grad():
        slice_drop, plane_drop = deletion_scores(model, mods)
    res = gradcam_all_labels(model, mods)
    res['slice_drop'], res['plane_drop'] = slice_drop, plane_drop

    if save_figs:
        os.makedirs(out_dir, exist_ok=True)
        for p in range(len(mods)):
            plot_cam_grid(res, p, f"{uid} | {cfg.glob.PLANES[p]} middle slice",
                          f"{out_dir}/{uid}_cam_{cfg.glob.PLANES[p]}.png", y_true)
        plot_importance(plane_drop, slice_drop, uid, f"{out_dir}/{uid}_importance.png")
    return res


# ─────────────────────────────────────────────
# 6. Loop over many studies + average results
#    loader: batch_size=1, yields (mods, meta, uids)
# ─────────────────────────────────────────────
def run_xai(model, loader, gt_df=None, out_dir='xai_out', n_study_figs=5, max_studies=None):
    os.makedirs(out_dir, exist_ok=True)
    gt = gt_df.set_index('StudyInstanceUID')[cfg.glob.LABELS] if gt_df is not None else None
    rec_plane, rec_prob = [], []
    slice_acc = [[] for _ in cfg.glob.PLANES]
    cam_sum = img_sum = None
    n = 0

    for batch in tqdm(loader, total=max_studies or len(loader)):
        if batch is None:
            continue
        mods, meta, uids = batch
        assert mods[0].shape[0] == 1, "use batch_size=1"
        uid    = uids[0]
        y_true = gt.loc[uid].values if (gt is not None and uid in gt.index) else None
        try:
            res = explain_study(model, mods, uid, y_true, out_dir,
                                save_figs=(n < n_study_figs), check=(n == 0))
        except torch.cuda.OutOfMemoryError:
            print(f"OOM on {uid}, skipped"); torch.cuda.empty_cache(); continue

        for p, pl in enumerate(cfg.glob.PLANES):
            for k, lb in enumerate(cfg.glob.LABELS):
                rec_plane.append(dict(uid=uid, plane=pl, label=lb, drop=res['plane_drop'][p, k]))
            slice_acc[p].append(res['slice_drop'][p])
        rec_prob.append(dict(uid=uid, **{lb: res['prob'][k] for k, lb in enumerate(cfg.glob.LABELS)},
                             **({f'GT_{lb}': y_true[k] for k, lb in enumerate(cfg.glob.LABELS)} if y_true is not None else {})))

        cam_sum = res['cam']['both'] if cam_sum is None else cam_sum + res['cam']['both']
        img_sum = res['imgs']        if img_sum is None else img_sum + res['imgs']
        n += 1
        torch.cuda.empty_cache()
        if max_studies and n >= max_studies:
            break

    # ---- save tables
    df_plane = pd.DataFrame(rec_plane)
    df_plane.to_csv(f"{out_dir}/plane_importance_per_study.csv", index=False)
    pd.DataFrame(rec_prob).to_csv(f"{out_dir}/probs.csv", index=False)
    mean_plane = df_plane.pivot_table(index='label', columns='plane', values='drop', aggfunc='mean') \
                         .reindex(index=cfg.glob.LABELS, columns=cfg.glob.PLANES)
    mean_plane.to_csv(f"{out_dir}/plane_importance_mean.csv")
    print("\nMean plane importance (logit drop):\n", mean_plane.round(3))

    # ---- average plots
    mean_slice = [np.mean(np.stack(a), axis=0) for a in slice_acc]       # (N, 12) per plane
    plot_importance(mean_plane.values.T, mean_slice, f"MEAN over {n} studies", f"{out_dir}/MEAN_importance.png")
    mean_res = dict(cam={'both': cam_sum / n}, imgs=img_sum / n,
                    prob=np.array([np.mean([r[lb] for r in rec_prob]) for lb in cfg.glob.LABELS]))
    for p in range(len(cfg.glob.PLANES)):
        plot_cam_grid(mean_res, p, f"MEAN Grad-CAM over {n} studies | {cfg.glob.PLANES[p]}",
                      f"{out_dir}/MEAN_cam_{cfg.glob.PLANES[p]}.png")
    print(f"\nDone. {n} studies. Files in {out_dir}/")
    return mean_plane