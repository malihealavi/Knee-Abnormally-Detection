# 🦵 RSNA Knee Abnormality Detection

A deep learning pipeline that reads knee MRI studies and predicts **12 knee problems** at the same time. Built for the RSNA Knee Abnormality Detection Kaggle competition.

**Metric:** macro ROC-AUC over the 12 targets.

| Group | Targets |
|---|---|
| Ligaments | ACL, MCL |
| Menisci | Medial Meniscus, Lateral Meniscus |
| Cartilage (OA) | Medial OA, Lateral OA, PF OA |
| Fluid / soft tissue | Effusion, Synovitis, Baker's cyst |
| Bone | Contusion, Fracture |

---

## 💡 The Main Challenge

Only **58 of 4,407 studies** have real (ground-truth) labels. The rest only have a text radiology report (in many languages).

**Our solution:**
1. Turn reports into labels with an LLM.
2. Pre-train image encoders without labels (self-supervised learning).
3. Train a multi-plane model, then add a small metadata model on top.

---

## 🗺️ Pipeline Overview

```mermaid
flowchart LR
    A[Radiology reports] -->|Llama-3 + rubric| B[Soft labels<br/>12 targets]
    C[DICOM series] --> D[Best series<br/>per plane]
    D --> E[16 slices<br/>256x256, 2.5D]
    D --> M[DICOM metadata]
    C -.->|no labels| S[SSL pre-training<br/>DINOv2 + ConvNeXt]
    S --> F
    E --> F[Dual-encoder<br/>fusion model]
    B --> F
    M --> G[LightGBM<br/>metadata model]
    F --> H[Blend in logit space]
    G --> H
    H --> I[12 probabilities]
```

---

## 🧹 Data Preprocessing

### 1. Label extraction from reports
`Preprocessing/label_extraction.ipynb`

- Model: **Llama-3-8B-Instruct** reads each report and gives each target a score.
- A strict rubric matches how the official labels work: only *moderate / high-grade / acute* findings are positive.
- Works for English, Spanish, German, Turkish, Greek, Dutch, Bulgarian and more.

| Score | Meaning |
|---|---|
| 1.0 | Clearly positive |
| 0.7 | Stated, but no severity given |
| 0.3 | Mentioned, but mild / below threshold |
| 0.1 | Not mentioned |
| 0.0 | Normal / negated |

Result: macro AUC on the 58 true-label studies went from about **0.80 to 0.85** after improving the rubric.

### 2. Series selection
`Preprocessing/light_df.py`

Each study has many series. For every plane (Axial, Coronal, Sagittal) we keep **one** series, in this order:
1. Fluid-sensitive **and** fat-suppressed
2. Fluid-sensitive only
3. Anything else

### 3. Image loading
`dataset/loader_3d.py`, `dataset/augmentation.py`, `dataset/stack_neighbor.py`

```mermaid
flowchart LR
    A[DICOM files] --> B[Take 16 middle slices<br/>VOI LUT]
    B --> C[Resize to 16x256x256<br/>normalize]
    C --> D[Augment<br/>train only]
    D --> E[2.5D stacking<br/>slice + 2 neighbors]
    E --> F[16 x 3 x 256 x 256<br/>per plane]
```

- **Augmentation (MONAI):** flips, rotation, zoom, noise, intensity shift/scale, bias field, Gibbs noise.
- **2.5D:** every slice is stacked with its previous and next slice as 3 channels, so a normal 2D network can see a little 3D context.
- Missing or broken planes are replaced by zeros, so one bad file never stops training.

### 4. Metadata features
`Preprocessing/feature_engineering.py`

33 features per plane (99 per study): scanner brand, field strength, sequence type, flip angle, image size, fluid / fat-suppression flags, and more. Numbers are scaled, missing values are set to `-1`.

---

## 🧠 Model Architecture

`model/MRI25DFusionModelViewRoutingDualEncoder.py`

```mermaid
flowchart TB
    subgraph PER["For each plane: Axial / Coronal / Sagittal"]
        direction TB
        X["16 slices (3 channels each)"] --> D1["DINOv2-small<br/>(SSL pre-trained)"]
        X --> C1["ConvNeXt-nano<br/>(SSL pre-trained)"]
        D1 --> P1["Linear to 256"]
        C1 --> P2["Linear to 256"]
        P1 --> CAT["Concat + Linear + LayerNorm"]
        P2 --> CAT
        CAT --> POOL["Slice Attention Pooler<br/>(transformer + CLS token)"]
    end
    POOL --> AX["Axial vector"]
    POOL --> CO["Coronal vector"]
    POOL --> SA["Sagittal vector"]
    AX --> FUS["Cross-Plane Fusion<br/>(transformer)"]
    CO --> FUS
    SA --> FUS
    AX --> H1["Axial head<br/>PF OA, Effusion, Synovitis"]
    FUS --> H1
    CO --> AVG["mean of<br/>Coronal + Sagittal"]
    SA --> AVG
    AVG --> H2["Coronal/Sagittal head<br/>ACL, MCL, Menisci, Medial/Lateral OA"]
    FUS --> H2
    AX --> ALL["mean of<br/>all planes"]
    CO --> ALL
    SA --> ALL
    ALL --> H3["General head<br/>Baker's, Contusion, Fracture"]
    FUS --> H3
    H1 --> OUT["12 logits"]
    H2 --> OUT
    H3 --> OUT
```

**Key ideas**

| Idea | Why |
|---|---|
| **Two encoders** (transformer + CNN) | They learn different things; combining them is stronger. |
| **Slice attention pooler** | Learns which slices matter and turns 16 slices into 1 vector. |
| **Cross-plane fusion** | Lets the three planes share information. |
| **View routing** | Each target uses the plane where doctors usually see it best. |

---

## 🎓 Self-Supervised Pre-training

`SSL/dino_ssl.py` (DINOv2-small) and `SSL/convnext_ssl.py` (ConvNeXt-nano)

- Method: DINO-style **self-distillation**. A student learns to match a slowly-updated teacher on different crops of the same slice. **No labels needed.**
- Uses all knee MRI slices from every series.
- Only the last blocks are trained, with a small learning rate.
- A **linear probe** on the 58 true-label studies checks if the features really improve (loss alone is not enough).

---

## 🏋️ Training

`train/Trainer.py`

| Setting | Value |
|---|---|
| Loss | BCE with logits (soft labels) |
| Optimizer | AdamW (backbone `1e-5`, heads `1e-4`) |
| Schedule | Cosine, 30 cycles × 3 data chunks |
| Backbones | Frozen first, then unfrozen step by step (last DINOv2 blocks / ConvNeXt stages) |
| Batch | Effective batch 128 with gradient accumulation |
| Precision | Mixed precision (AMP) |
| Stopping | Early stopping (patience 6) |
| Validation | 10% held-out split **and** the 58 true-label studies |

---

## 🧾 Metadata Model and Blending

`Trainer.meta_trainer()` and `utils/utils.py`

- Metadata alone is weak (**about 0.64 val AUC**) but adds a different signal.
- Logistic Regression, LightGBM and HistGradientBoosting are tested. The best one on validation is used (LightGBM).
- The final prediction mixes both in logit space:

```
final = sigmoid( logit(image_prob) + w * logit(metadata_prob) )
```

- The weight `w` is chosen on validation with bootstrap checks. If the gain is not reliable, `w = 0` (image model only).
- A per-label version (`fit_label_weights`) is also available.

---

## 🔍 Explainability (XAI)

`XAI/xai.py`, results in `assets/XAI/xai_out/`

Three tools show *why* the model answers the way it does:

1. **Plane deletion:** hide one plane and see how much the score drops.
2. **Slice deletion:** hide one slice at a time.
3. **Grad-CAM:** heat maps from both encoders.

**What we found (mean over 58 true-label studies):**

| Target | Most important plane(s) |
|---|---|
| ACL | Sagittal, Axial |
| Medial / Lateral OA | Coronal |
| Medial Meniscus | Coronal |
| Synovitis, Effusion | Axial |
| Contusion, Fracture | Sagittal, Axial |

This matches how radiologists read knee MRI, which is a good sign the model learns real patterns.

### Detailed XAI results (58 true-label studies)

All numbers are averages over the 58 studies that have real labels.

#### 1. Plane deletion

We hide one whole plane and measure the change of the model score (in logit units, a stretched version of probability).
**Positive = the plane supports the target. Negative = the plane pulls the score down** (hiding it makes the score go up, so it may add noise).

| Target | Axial | Coronal | Sagittal |
|---|---|---|---|
| ACL | 0.24 | **-0.48** | **0.41** |
| MCL | 0.06 | -0.21 | -0.04 |
| Medial Meniscus | 0.17 | **0.37** | -0.07 |
| Lateral Meniscus | 0.20 | 0.10 | -0.22 |
| Medial OA | 0.15 | **0.51** | -0.11 |
| Lateral OA | 0.11 | **0.41** | -0.15 |
| PF OA | -0.09 | 0.12 | -0.05 |
| Effusion | 0.18 | 0.08 | 0.04 |
| Synovitis | **0.40** | 0.05 | 0.00 |
| Baker's | 0.25 | -0.13 | 0.09 |
| Contusion | 0.10 | **-0.39** | **0.39** |
| Fracture | **0.46** | **-0.37** | **0.44** |

![Mean importance](assets/XAI/xai_out/MEAN_importance.png)

**Main points**

- **The view routing works.** The plane the model trusts most is the plane doctors use: Sagittal for ACL, Coronal for menisci and OA, Axial for Synovitis and Baker's cyst.
- **The Coronal plane hurts some targets.** For ACL, Contusion and Fracture, the Coronal plane has a clearly negative effect (-0.37 to -0.48). The model seems to use it as noise there. Hiding it for these targets could help (needs a validation test).
- **Fracture has the strongest signals.** Axial (0.46) and Sagittal (0.44) both help. Fracture is also the only target where all the effect sizes are large.
- **Weak targets:** MCL has almost no helpful plane (all values near zero or negative), and PF OA is weak in every plane. Lateral Meniscus gets help from Axial only; Sagittal works against it (-0.22).
- **Effusion is spread over all three planes** (small positive values in each), so no single plane is key.
- Per-study check: ACL, Fracture and Contusion are supported by the Sagittal plane in 86-90% of studies, and Medial Meniscus by the Axial and Coronal planes in 81-88%. So the average is not driven by a few studies.

#### 2. Slice deletion

The three right-hand panels of the figure above show the effect of hiding one slice (index 0-15 out of the 16 middle slices).

- Only a **few slices matter** for each target, others are close to zero. The slice pooler learned to focus.
- **Coronal:** slices 3-5, 8 and 12 are the most useful for ligament, meniscus and OA targets.
- **Sagittal:** slices 7-8 matter for ACL, Medial Meniscus and Fracture (the center of the knee, where the ACL and menisci are).
- **Axial:** the first slices (0-1) are strongly positive for Effusion and Synovitis.
- Many slices have a strong **negative** effect (for example Axial slice 6 for Effusion, Synovitis and Fracture). This means some slices push the score the wrong way, which is a sign of noise and a possible place to improve (for example, slice dropout during training).

#### 3. Grad-CAM (where the model looks)

Heat maps for the middle slice, averaged over 58 studies (bright = more important). Because they are averages of many different knees, the images look blurry, so only the main area should be read.

![Axial](assets/XAI/xai_out/MEAN_cam_Axial.png)
![Coronal](assets/XAI/xai_out/MEAN_cam_Coronal.png)
![Sagittal](assets/XAI/xai_out/MEAN_cam_Sagittal.png)

| Plane | Target | Where the model looks | Makes sense? |
|---|---|---|---|
| Sagittal | ACL, Medial Meniscus, Medial/Lateral OA | Center of the joint, between femur and tibia | ✅ Yes, this is where the ACL and menisci sit |
| Sagittal | Contusion | Femur bone (upper part) | ✅ Yes, bone bruise is in the bone |
| Coronal | Medial/Lateral Meniscus, OA | A horizontal band along the joint line | ✅ Yes, menisci and cartilage lie here |
| Coronal | ACL, MCL | Left side of the joint and the center | ✅ Partly (side ligaments and center) |
| Coronal | Contusion | Bone on the left side | ✅ Likely |
| Axial | Effusion, Synovitis | Left and right sides of the joint | ✅ Yes, fluid and inflamed lining collect around the joint |
| Axial | Baker's | Back (posterior) center | ✅ Yes, Baker's cyst is behind the knee |
| Axial | PF OA | Back (posterior) center | ⚠️ No, the patella is at the front. Likely a weak or shortcut signal |

#### 4. How to read this

- **Good news:** for most targets (ACL, menisci, OA, Effusion, Synovitis, Baker's) the model looks at the right body part in the right plane, and the plane importance matches clinical practice.
- **Weak spots:** MCL, PF OA and Medial Meniscus. On the 58 true-label studies their AUC is lowest (about 0.71, 0.84 and 0.59), and their XAI signals are weak or odd. Medial Meniscus is used a lot by the model (Coronal 0.37) but still ranks cases poorly, so "the model looks there" does not always mean "the model is right".
- **Ideas to try next:** (1) ignore the Coronal plane for ACL, Contusion and Fracture, (2) add slice dropout in training, (3) use more series per plane for MCL and PF OA.
- **Limits:** Grad-CAM uses only the middle slice, the maps are normalized per target, and 58 studies is a small sample. Deletion tests show what the model *uses*, not what is medically *true*.

---

## 📁 Repository Structure

```
├── main.py                  # Train + run XAI
├── configs.py               # All settings and data paths
├── dataset/
│   ├── KneeDataset.py       # Study-level dataset (3 planes + labels)
│   ├── SafeSubmissionDataset.py
│   ├── loader_3d.py         # DICOM loading
│   ├── augmentation.py      # MONAI transforms
│   └── stack_neighbor.py    # 2.5D stacking
├── model/
│   ├── MRI25DFusionModelViewRoutingDualEncoder.py
│   ├── SliceEncoder.py
│   ├── SliceAttentionPooler.py
│   ├── CrossModalFusion.py
│   └── MetaEncoder.py
├── SSL/                     # Self-supervised pre-training
├── Preprocessing/           # Labels, series selection, metadata
├── train/Trainer.py         # Training, validation, blending
├── utils/utils.py           # Metrics, blending, collate, memory tools
├── XAI/                     # Explainability code
├── notebook/                # Kaggle notebooks
└── assets/                  # Data, checkpoints, XAI outputs
```

---

## 🚀 Quick Start

```bash
pip install torch timm transformers monai==1.4.0 pydicom \
            lightgbm scikit-learn pandas numpy matplotlib tqdm psutil
```

1. Put the CSV files in `assets/data/` (see `configs.py` for names).
2. Run the label extraction notebook, SSL scripts, and metadata preprocessing.
3. Train and explain:

```bash
python main.py
```

> **Note:** Some paths (SSL weights, DICOM folders) are set for Kaggle (`/kaggle/input/...`). Change them for your own setup.

---

## 📌 Results Summary

| Item | Result |
|---|---|
| Report labels vs true labels (58 studies) | macro AUC ≈ 0.849 |
| Metadata model only | ≈ 0.64 (val) |
| Public leaderboard (dinov2 model only) | ≈ 0.795 |
| Private leaderboard (dinov2 model only) | ≈ 0.803 |
| Public leaderboard (dinov2 model + weighted(0.2) meta ) | ≈ 0.786 |
| Private leaderboard (dinov2 model + weighted (0.2) meta) | ≈ 0.794 |
| Public leaderboard (dinov2 model + weighted(0.3) meta ) | ≈ 0.795 |
| Private leaderboard (dinov2 model + weighted (0.3) meta) | ≈ 0.803 |
| Public leaderboard (dual encoder) | ≈ 0.813 |
| Private leaderboard (dual encoder) | ≈ 0.819 |
| Public leaderboard (ensemble: dual encoder + dinov2 + conv_nano ) | ≈ 0.815 |
| Private leaderboard (ensemble: dual encoder+ dinov2 + conv_nano) | ≈ 0.819 |
---

## 🔮 Future Work

- Use all series per plane, not just one.
- Higher resolution (e.g. 384 px) in later training stages.
- Better rubric rules for hard targets (OA, Baker's cyst).

---

## 🙏 Acknowledgements

RSNA and Kaggle for the competition and data. DINOv2 (Meta), ConvNeXt (timm), MONAI, Llama-3.
