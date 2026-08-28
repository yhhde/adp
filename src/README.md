# src/ — training code

Two trainers plus one batch runner. Everything here runs under the `adp`
conda environment (see the repo README) and needs CUDA for a real run.

| File | Role |
|------|------|
| `train_prompt.py`   | Prompt method — ProFiT-style MLM verbalizer (masked LM + verbalizer logits) |
| `train_sequence.py` | Sequence method — standard classification head on the `[CLS]`/pooler output |
| `common_utils.py`   | Shared infrastructure: dataset loading, deterministic seeding, unified logging/checkpoint helpers, prompt templates & verbalizers |
| `run_experiments.py`| Batch runner: reads `configs/experiments.jsonl`, launches one trainer per (config, seed) pair |

## Protocol

These are the design choices of the experiment, uniform across both methods:

- **Class weights** — balanced weights computed from the training set
  (sklearn `compute_class_weight('balanced')`), used in the loss.
- **Early stopping** — prompt: on, patience 5, applied unconditionally.
  sequence: off (runs the full epoch budget). The runner never passes
  `--early_stopping` to `train_sequence.py`, so this is the default.
- **Evaluation** — the **final in-memory model** (last weights) is evaluated
  on val and test. The best-val checkpoint is archived under
  `experiments/<id>/best_model` but never reloaded for evaluation.
- **Determinism** — seeds are set through `set_random_seed` in
  `common_utils.py` (torch, numpy, random, plus deterministic cuDNN
  algorithms); the worker scripts also pin `PYTHONHASHSEED` and
  `CUBLAS_WORKSPACE_CONFIG`.

## Running one experiment (single config line)

```bash
python src/run_experiments.py --config configs/experiments.jsonl --index 0 \
    --device 0 --output-dir runs_local
```

Or invoke a trainer directly, e.g.:

```bash
python src/train_prompt.py --model_id bert-base-multilingual-uncased \
    --data_path data --lr 4e-5 --train_batch_size 8 \
    --gradient_accumulation_steps 16 --num_epochs 15 --prompt_pattern 1
```

## Batch mode

`run_experiments.py` reads one JSONL line per experiment and launches each
(config, seed) pair as a subprocess:

- `--seeds S1,S2` … comma-separated list (serial sweep, e.g. `run_all.sh`)
- `--index I --seed S` … single pair (per-job grid, e.g. `adp.sub`)

A skip-guard manifest at `<output-dir>/run_manifest.jsonl` records the exit
status of each pair; pairs already marked `ok` are skipped on rerun, so an
interrupted batch can be resubmitted safely.

All trainer output (logs, checkpoints, summaries) lands under `--output-dir`.
