# SLT Evaluation

Evaluate trained SLT checkpoints on `dev` and `test` for:

- `main_pipeline`
- `baseline_s1`
- `baseline_s3`

Metrics:

- BLEU-1
- BLEU-2
- BLEU-3
- BLEU-4
- ROUGE-1
- ROUGE-2
- ROUGE-L

Run on SLURM:

```bash
sbatch eval_slt/job_eval_slt.sh
```

Run directly:

```bash
python -u eval_slt/eval_slt.py --config eval_slt/config_eval_slt.yaml
```

Evaluate only the GT-oracle checkpoint:

```bash
sbatch --export=ALL,CONFIG=eval_slt/config_eval_gt_oracle.yaml eval_slt/job_eval_slt.sh
```

Outputs are written to:

```text
output/eval_slt/
```

Important files:

- `metrics.txt`: human-readable summary with all BLEU/ROUGE metrics
- `metrics.json`: full raw values
- `predictions_<model>_<split>.txt`: hypotheses and references
