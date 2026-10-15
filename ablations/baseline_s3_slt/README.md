# Baseline S3-only CSLR -> SLT

This ablation is isolated from the main training jobs.

## Architecture

```text
Input video
  -> Stream 3 Cross-Attention CSLR
  -> CTC decode predicted gloss
  +  Stream 3 visual features
  -> Late fusion
  -> mBART-50 SLT decoder
  -> Translation sentence
```

## Default Checkpoint

```text
checkpoints/stream3_only/best_stream3.pth
```

## Gloss Source

The default config trains SLT with predicted Stream 3 gloss to match evaluation:

```yaml
train_gloss_source: "predicted"
```

Evaluation uses predicted Stream 3 CTC gloss to represent the full baseline pipeline:

```yaml
eval_gloss_source: "predicted"
```

## Run

From the project root:

```bash
sbatch ablations/baseline_s3_slt/job_baseline_s3_slt.sh
```

## Outputs

```text
checkpoints/ablations/baseline_s3_slt/best_baseline_s3_slt.pth
logs/ablations/baseline_s3_slt/
output/output_baseline_s3_slt.txt
output/out_baseline_s3_slt.err
```
