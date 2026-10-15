# Visual Feature Ablation

This ablation keeps the main pipeline training setup but disables visual features inside the SLT stage.

The CSLR streams still run to produce the predicted weighted-ROVER gloss. After the predicted gloss is selected, the SLT fusion module receives zero visual features, so the decoder must rely on the predicted gloss pathway instead of visual content.

## Train No-Visual Variant

From the project root:

```bash
python -u visual_feature_ablation/train_main_novf.py \
  --config visual_feature_ablation/config_no_visual.yaml
```

On SLURM:

```bash
sbatch visual_feature_ablation/job_train_no_visual.sh
```

## Compare With Main Pipeline

After the no-visual checkpoint exists:

```bash
python -u visual_feature_ablation/eval_visual_feature.py \
  --config visual_feature_ablation/config_eval_visual_feature.yaml
```

On SLURM:

```bash
sbatch visual_feature_ablation/job_eval_visual_feature.sh
```

## Outputs

Training:

```text
checkpoints/visual_feature_ablation/no_visual/best_main_pipeline_slt.pth
logs/visual_feature_ablation/no_visual/
```

Evaluation:

```text
output/visual_feature_ablation/metrics.txt
output/visual_feature_ablation/metrics.csv
output/visual_feature_ablation/metrics_delta.csv
output/visual_feature_ablation/metrics.json
output/visual_feature_ablation/predictions_<variant>_<split>.txt
```
