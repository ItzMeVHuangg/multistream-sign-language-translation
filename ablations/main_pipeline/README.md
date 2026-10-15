# Main Pipeline: Weighted ROVER CSLR -> SLT

This folder contains the agreed final pipeline, isolated from the main jobs.

## Architecture

```text
Input video
  -> Stream 1 RGB-USTM              -> CTC hypothesis S1
  -> Stream 2 Hypergraph Skeleton   -> CTC hypothesis S2
  -> Stream 3 Cross-Attention       -> CTC hypothesis S3 + s3_feats
  -> Weighted ROVER [1.0, 0.3, 1.0]
  -> Final gloss sequence
  +  Stream 3 visual features
  -> Late fusion
  -> mBART-50 SLT decoder
  -> Translation sentence
```

## Checkpoints

```text
checkpoints/stream1_only/best_stream1.pth
checkpoints/stream2_only/best_stream2.pth
checkpoints/stream3_only/best_stream3.pth
```

## Gloss Source

Training uses predicted weighted ROVER gloss to match the deployed input distribution:

```yaml
train_gloss_source: "rover"
```

Evaluation uses predicted weighted ROVER gloss:

```yaml
eval_gloss_source: "rover"
```

## Run

From the project root:

```bash
sbatch ablations/main_pipeline/job_main_pipeline.sh
```

## Outputs

```text
checkpoints/ablations/main_pipeline/best_main_pipeline_slt.pth
logs/ablations/main_pipeline/
output/output_main_pipeline.txt
output/out_main_pipeline.err
```
