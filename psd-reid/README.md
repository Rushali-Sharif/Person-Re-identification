# Platonic Self-Distillation for Person Re-Identification

Supervised person re-ID is close to saturated in-domain, yet the same models collapse
when moved to a camera network they were not trained on — source-only transfer to
MSMT17 is reported at 5.2 mAP. The usual remedy, clustering the unlabelled target into
pseudo-labels, has a specific failure mode: domain shift produces errors that are
*systematic*, and the reliability checks in common use (clustering consistency,
neighbourhood consistency, teacher–student agreement) all read the model's own state,
so a systematic error looks self-consistent and gets marked reliable.

**Platonic Self-Distillation (PSD)** adds evidence from outside the model. Motivated by
the Platonic Representation Hypothesis — independently trained models converge on shared
structure but have no reason to share each other's *private* errors — each image's
distillation target is weighted by how far a frozen, independently trained referee
corroborates the teacher's neighbourhood for that image.

A frozen general-purpose referee does not degrade as the target domain changes, whereas
a source-trained teacher does, so the gate should matter more as the domain gap widens.

PSD changes the training objective only. The backbone, the forward pass and the
inference path are untouched, so it adds **no cost at test time** and composes with any
re-ID architecture rather than competing with it.

## Method

```
                       ┌──────────────────────┐
                       │ neighbour agreement  │
                       │      gate  w_i       │
                       └───▲──────────────▲───┘
                           │              │
              ┌────────────┴───┐   ┌──────┴────────┐
              │  EMA teacher   │   │ DINOv2 referee│
              │  feature bank  │   │   (frozen)    │
              └────────▲───────┘   └───────────────┘
                   EMA │
  image ──▶ student (CLIP ViT-B/16 + BNNeck) ──▶ L_ID + L_tri + λ·w_i·(1 − cos(s_i, t_i))
```

Per epoch: the teacher encodes the training set into a bank; for each image the teacher's
*k*=20 nearest neighbours are compared against the frozen referee's; the chance-corrected
overlap gives a gate `w_i ∈ [0,1]`; the distillation term is scaled by `w_i`. Images the
referee does not corroborate contribute almost no gradient.

## Results

Market-1501, single query, no re-ranking, one seed, 40 epochs on a single T4.

| Variant | mAP | R-1 | R-5 | R-10 |
|---|---|---|---|---|
| Baseline (ID + triplet) | 89.17 | 94.83 | 98.69 | 99.38 |
| **PSD, iteration 1** (gated feature distillation) | **89.50** | **95.37** | 98.55 | 99.32 |
| + relation loss, gate disabled | 86.65 | 94.09 | 98.10 | 98.78 |
| + relation loss, gate enabled | 14.44 | 28.92 | 39.96 | 45.90 |

Cross-dataset, Market-1501 → CUHK03, direct transfer with no adaptation:

| Model | Split | mAP | R-1 | R-5 | R-10 |
|---|---|---|---|---|---|
| PSD, iteration 1 | CUHK03 detected | 31.50 | 32.07 | 52.50 | 61.71 |
| PSD, iteration 1 | CUHK03 labeled | 32.43 | 33.64 | 53.86 | 63.86 |

**Read these honestly.** The in-domain gain of +0.33 mAP comes from a single seed and is
the size of normal run-to-run variation; it is reported as unconfirmed. The cross-dataset
row has no baseline to compare against yet, so the central claim — that PSD's margin
*widens* under domain shift — remains untested.

**Known limitation.** The DINOv2 referee is a weak judge of identity (reported at 1.71%
mAP on Market-1501 zero-shot), which is the likely reason the gate sits at a mean weight
of ≈0.14 with ~7% of neighbours confirmed, flat across all 40 epochs. The mechanism and
the choice of referee are separate decisions; on this evidence the referee is the wrong
one. Replacing it with an identity-discriminative encoder is the first item of the next
iteration.

## Layout

```
psd_reid.py                   training and evaluation (baseline and PSD in one script)
scripts/eval_cuhk03.py        Market → CUHK03 direct transfer (767/700 new protocol)
scripts/eval_corruption.py    corruption-robustness sweep on Market-1501
scripts/build_hf_space.py     package a trained checkpoint into a Hugging Face Space
app/app_kaggle.py             Gradio retrieval + video-tracing application
app/prw_video.py              render an annotated cross-camera tracking video
notebooks/                    Kaggle notebook for the full pipeline
```

## Usage

```bash
pip install -r requirements.txt

# baseline
python psd_reid.py --root /path/to/Market-1501 --name base --epochs 40

# DINOv2 referee features (once)
python psd_reid.py --root /path/to/Market-1501 --make-dino dino_train.pt

# PSD, iteration 1
python psd_reid.py --root /path/to/Market-1501 --name psd_iter1 \
                   --psd --lam-rel 0 --dino dino_train.pt --epochs 40
```

Checkpoints and per-epoch logs are written to `psd_reid_out/`.

## Demo application

`app/app_kaggle.py` launches a Gradio interface with two modes: image-query retrieval
over an indexed gallery, and tracing a queried person through video frames with bounding
boxes and per-detection similarity scores. `app/prw_video.py` renders the same as a video
file.

## Data and checkpoints

Neither datasets nor trained weights are included.

- **Market-1501** and **CUHK03** must be obtained from their original distributors under
  their own terms.
- **PRW**, used only by the demo for full video frames, is licensed for research use. It
  contains identifiable people filmed without consent to appear in a public search tool,
  so do not redistribute its frames or deploy a publicly accessible instance of the demo
  over them.

## Acknowledgements

Builds on CLIP, DINOv2, the BNNeck and batch-hard triplet recipe from *Bag of Tricks*,
and the mean-teacher formulation. The motivating hypothesis is from Huh et al.,
*Position: The Platonic Representation Hypothesis* (ICML 2024).
