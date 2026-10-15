# ROVER Weight Ablations

This folder evaluates ROVER voting weights without changing the main config.

## Cases

```text
no_stream2    : stream_weights = [1.0, 0.0, 1.0]
equal_voting  : stream_weights = [1.0, 1.0, 1.0]
```

Both cases use:

```text
beam_width = 1
use_lm = false
decode = greedy
```

## Run

From the project root:

```bash
sbatch ablations/rover_weights/job_eval_rover_weights.sh
```

## Output

```text
output/output_ablation_rover_weights.txt
output/out_ablation_rover_weights.err
```

Use the reported DEV/TEST ROVER WER values in the ROVER weight ablation table.
