# Baseline S1-only CSLR -> SLT

This ablation evaluates SLT with Stream 1 only.

## Architecture

```text
Input video
  -> Stream 1 RGB-USTM
  -> CTC decode predicted gloss
  +  Stream 1 visual features
  -> Late fusion
  -> mBART-50 SLT decoder
  -> Translation sentence
```

## Default Checkpoint

```text
checkpoints/stream1_only/best_stream1.pth
```

## Gloss Source

Training uses predicted Stream 1 gloss to match evaluation:

```yaml
train_gloss_source: "predicted"
```

Evaluation uses predicted Stream 1 CTC gloss:

```yaml
eval_gloss_source: "predicted"
```

## Run

From the project root:

```bash
sbatch ablations/baseline_s1_slt/job_baseline_s1_slt.sh
```

## Outputs

```text
checkpoints/ablations/baseline_s1_slt/best_baseline_s1_slt.pth
logs/ablations/baseline_s1_slt/
output/output_baseline_s1_slt.txt
output/out_baseline_s1_slt.err
```
