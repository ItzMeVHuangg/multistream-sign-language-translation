# Multi-Stream Continuous Sign Language Recognition & Translation

> Video → Gloss → German text on **PHOENIX-2014T**, using three CTC recognition streams (RGB, skeleton, RGB×keypoint cross-attention), a **Weighted ROVER** ensemble, and an **mBART-50** translator.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="HuggingFace" src="https://img.shields.io/badge/🤗%20Transformers-mBART--50-FFD21E">
  <img alt="MediaPipe" src="https://img.shields.io/badge/MediaPipe-Keypoints-0097A7">
  <img alt="Dataset" src="https://img.shields.io/badge/Dataset-PHOENIX--2014T-6A5ACD">
</p>

This project was built as an undergraduate thesis at the **University of Information Technology, VNU-HCM** (Computer Science, 2026):
*"Continuous Sign Language Recognition and Translation from Video to Text using Deep Learning"*.

---

## Highlights

- **Three complementary CSLR streams**, each trained with CTC loss:
  1. **RGB-USTM (adapted):** a Video Swin-T (`swin3d_t`) backbone with Temporal-Aware Positional Encoding (TAPE) and temporal refinement.
  2. **Hypergraph Skeleton:** a HyperSign-lite model over 75 MediaPipe joints (pose + both hands), using KNN geometric hyperedges, semantic prototype hyperedges and multi-scale temporal convolutions.
  3. **Cross-Attention Fusion:** RGB features attend to keypoint features, followed by a BiLSTM. This is the best single stream.
- **Weighted ROVER ensemble:** aligns the three gloss hypotheses and runs a weighted vote at sequence level. The weights `[1.0, 0.3, 1.0]` down-weight the weaker skeleton stream.
- **Gloss + visual SLT:** the ROVER gloss is late-fused with Stream-3 visual features and decoded by **mBART-50** using a visual prefix and adapter. Only some modules are fine-tuned.
- **Ablation studies:** single-stream baselines, ROVER weight variants, the SLT model without visual features, and a ground-truth gloss oracle.

## Architecture

```text
                   ┌─► Stream 1: RGB-USTM (Swin3D-T + TAPE)        ─► CTC gloss S1 ─┐
Input video ───────┼─► Stream 2: Hypergraph Skeleton (MediaPipe)   ─► CTC gloss S2 ─┼─► Weighted ROVER ─► Final gloss
 (+ keypoints)     └─► Stream 3: Cross-Attention RGB × Keypoint    ─► CTC gloss S3 ─┘   [1.0, 0.3, 1.0]       │
                                     │                                                                        │
                                     └──────────── Stream-3 visual features ──────────► Late fusion ◄─────────┘
                                                                                             │
                                                                                     mBART-50 decoder
                                                                                             │
                                                                                    German sentence
```

## Results (PHOENIX-2014T)

### Stage 1: CSLR (WER ↓)

| Model | Dev | Test |
|---|---|---|
| Stream 1: RGB-USTM | 28.07 | 27.91 |
| Stream 2: Hypergraph Skeleton | 57.98 | 57.43 |
| Stream 3: Cross-Attention Fusion | **25.72** | 27.27 |
| Joint CSLR (fuse 3 streams before CTC decoding) | 30.46 | 32.60 |
| ROVER S1 + S3, weights `[1.0, 0.0, 1.0]` | 28.79 | 28.87 |
| **Weighted ROVER `[1.0, 0.3, 1.0]`** | 27.77 | **26.99** |
| Weighted ROVER `[1.0, 0.5, 1.0]` | 28.21 | 28.01 |
| Equal ROVER `[1.0, 1.0, 1.0]` | 28.05 | 27.91 |

### Stage 2: SLT (test set, by input gloss source)

| Input gloss | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L |
|---|---|---|---|---|---|
| Stream 1 | 28.64 | 18.76 | 13.02 | 9.78 | 27.27 |
| Stream 3 | 29.04 | 19.26 | 15.35 | 10.49 | 27.79 |
| **Weighted ROVER** | **37.61** | **26.07** | **20.39** | **16.53** | **34.05** |
| Ground-truth gloss (oracle) | 49.76 | 38.09 | 30.20 | 24.91 | 48.86 |

On the dev set, the ROVER gloss reaches **BLEU-4 17.64 / ROUGE-L 37.92**. With ground-truth gloss, the translator scores close to T-MORT (BLEU-4 25.07). The gap between the predicted and oracle gloss shows that recognition errors carry over into translation, and this remains the main bottleneck.

## Project Structure

```text
├── configs/config.yaml              # All hyperparameters and paths
├── data/dataset.py                  # PHOENIX-2014T dataset, vocabularies, transforms
├── models/
│   ├── stream1_rgb_ustm.py          # Swin3D-T + TAPE
│   ├── stream2_hypergraph.py        # Hypergraph skeleton encoder
│   ├── stream3_crossattn.py         # RGB × keypoint cross-attention + BiLSTM
│   ├── rover_ensemble.py            # Weighted ROVER alignment and voting
│   ├── late_fusion.py               # Gloss + visual feature fusion
│   └── translator_llm.py            # mBART-50 translator with visual prefix
├── training/
│   ├── train_single_stream.py       # Train stream 1 / 2 / 3 on its own
│   ├── train_cslr.py                # Joint three-stream CSLR
│   ├── eval_three_stream.py         # Evaluate streams + ROVER
│   └── train_slt.py                 # Train the SLT stage
├── ablations/                       # Main pipeline, S1/S3 baselines, ROVER weight ablations
├── visual_feature_ablation/         # SLT without visual features
├── eval_slt/                        # BLEU / ROUGE evaluation for SLT checkpoints
├── compare_generated_sentences/     # Qualitative samples (gloss and sentences)
├── utils/                           # CTC decoder, WER/BLEU/ROUGE metrics, logger
└── extract_mediapipe_fast.py        # MediaPipe pose + hand keypoint extraction
```

## Getting Started

### 1. Environment

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
pip install mediapipe opencv-python
```

### 2. Data

1. Download [RWTH-PHOENIX-Weather 2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/).
2. Set `paths.*` in [configs/config.yaml](configs/config.yaml) to your local dataset folder.
3. Download the MediaPipe models [`hand_landmarker.task`](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) and [`pose_landmarker_lite.task`](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task) into the repo root, then extract keypoints:
   ```bash
   python extract_mediapipe_fast.py
   ```

### 3. Training

```bash
# Stage 1: train each CSLR stream
python training/train_single_stream.py --config configs/config.yaml --stream 1
python training/train_single_stream.py --config configs/config.yaml --stream 2
python training/train_single_stream.py --config configs/config.yaml --stream 3

# Evaluate the streams and the Weighted ROVER ensemble
python training/eval_three_stream.py --split test --decode greedy --save_hyps

# Stage 2: SLT with ROVER gloss + visual features (main pipeline)
python ablations/main_pipeline/train_main_pipeline.py --config ablations/main_pipeline/config_main_pipeline.yaml
```

Training ran on an NVIDIA DGX A100 (80 GB). For smaller GPUs, lower `batch_size` and raise `grad_accumulation_steps` in the config.

### 4. Evaluation

```bash
python eval_slt/eval_slt.py --config eval_slt/config_eval_slt.yaml
```

Each ablation folder has its own README with instructions for that experiment. Its `sbatch` commands are for SLURM; to run locally, call the `python` command directly.

## Tech Stack

PyTorch · torchvision (Video Swin) · HuggingFace Transformers (mBART-50) · MediaPipe · CTC · NLTK / ROUGE · TensorBoard · SLURM on an NVIDIA DGX A100

## Authors

- **Vũ Việt Hoàng** - **23520548**
- **Mai Thái Bình** - **23520158**

Faculty of Computer Science, University of Information Technology (UIT), VNU-HCM
Supervisor: **MSc. Cáp Phạm Đình Thăng**

## Acknowledgements

- [PHOENIX-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) dataset (Camgoz et al., CVPR 2018)
- USTM (Hasanaath & Luqman) and HyperSign, which inspired Streams 1 and 2
- [mBART-50](https://huggingface.co/facebook/mbart-large-50) from Meta AI
