# Generated Sentence Comparison

Exports qualitative samples from the trained main pipeline on the test split.

Each sample contains:

- reference gloss
- predicted ROVER gloss
- reference German sentence
- generated German sentence

## Run

From the project root:

```bash
python -u compare_generated_sentences/compare_generated_sentences.py \
  --config compare_generated_sentences/config_compare_generated_sentences.yaml
```

On SLURM:

```bash
sbatch compare_generated_sentences/job_compare_generated_sentences.sh
```

## Output

```text
output/compare_generated_sentences/comparison_samples.md
output/compare_generated_sentences/comparison_samples.csv
output/compare_generated_sentences/comparison_samples.jsonl
```
