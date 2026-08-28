# ADP — Multilingual Alzheimer's Disease Prediction

Multilingual text classification into `NC` / `MCI` / `pAD`, with two methods:
vanilla sequence fine-tuning and prompt-based fine-tuning (MLM verbalizer).
Training entry point is `src/run_experiments.py` (one JSONL line = one
experiment, looped over seeds); single runs go directly to
`src/train_prompt.py` / `src/train_sequence.py`.

`configs/experiments.jsonl` contains an example set of configurations — see
§4 for the config format and §6 for how to launch a batch.

## 1. Project structure

```text
adp-main/
├── src/
│   ├── train_prompt.py          # Prompt method (ProFiT-style MLM verbalizer)
│   ├── train_sequence.py        # Sequence method (classification head)
│   ├── common_utils.py          # Shared infra (dirs, logging, checkpoints)
│   └── run_experiments.py       # Batch runner: JSONL configs × seeds
├── configs/
│   └── experiments.jsonl        # Example config set
├── data/                        # train.csv / val.csv / test.csv
├── adp.sub                      # HTCondor: one job per (config, seed) pair
├── adp_batch.sub                # HTCondor: whole sweep on one node (serial)
├── run_adp.sh                   # Worker script for adp.sub
├── run_all.sh                   # Worker script for adp_batch.sub
└── rsync_adp.sh                 # Sync this repo to the cluster
```

## 2. Environment

```bash
conda create -n adp python=3.10 -y
conda activate adp
pip install -r requirements.txt
```

If the cluster provides PyTorch/CUDA modules, load them first, then install the
rest.

## 3. Data

`data/` must contain `train.csv`, `val.csv`, `test.csv` with columns:

| Column | Notes |
|--------|-------|
| `text` | input transcript |
| `label` | one of `NC`, `MCI`, `pAD` |
| `lang` | optional; used for per-language filtering/metrics |

Current splits: 804 train / 113 val / 102 test (en/zh/el).

## 4. Config files

`configs/experiments.jsonl` is an example config set — one experiment per
line, JSON. The batch runner reads it line by line. Supported fields:

| Field | Meaning |
|-------|---------|
| `method` | `prompt` or `sequence` |
| `model_id` | HuggingFace model name, e.g. `bert-base-multilingual-uncased` |
| `lr`, `train_batch_size`, `eval_batch_size`, `gradient_accumulation_steps`, `num_epochs`, `max_length`, `warmup_ratio`, `weight_decay`, `early_stopping_patience` | hyperparameters |
| `prompt_pattern` | prompt method only; template 1–4 from `src/common_utils.py` |
| `class_weights` | `true` = balanced class weights computed from the training set |

To run a different set of experiments, write your own JSONL file and point
`--config` (or the submit file's `arguments`) at it.

## 5. Running locally (single config line)

```bash
python src/run_experiments.py --config configs/experiments.jsonl --index 0 \
    --device 0 --output-dir runs_local
```

The trainers require CUDA (pass `--force_cpu` to a trainer for a CPU-only
run); production jobs must run on GPU.

## 6. Running a batch on the cluster (HTCondor)

Two submit files with different granularity — pick one:

| File | Worker | Granularity | Use when |
|------|--------|-------------|----------|
| `adp_batch.sub` | `run_all.sh` | 1 job = 1 node, every config run serially | account limited to one concurrent job; configs never wait on the scheduler |
| `adp.sub` | `run_adp.sh` | 1 job = 1 (config, seed) pair | several GPUs available; configs run in parallel |

Both pass `config output_dir seeds [device]` to their worker via the
`arguments` line, and share the same log naming (`condor_logs/adp_*`),
resources (1 GPU / 8 CPUs / 32GB) and the same output-dir convention — so a
sweep can be launched with either, and the skip-guard manifest carries over
(see below).

```bash
# 1. Sync the repo to the cluster
bash rsync_adp.sh            # dry-run first; add --go to sync for real

# 2. Submit the sweep
condor_submit adp_batch.sub  # serial sweep on one node (recommended)
# or
condor_submit adp.sub        # per-(config, seed) parallel jobs

# 3. Monitor
condor_q -u $USER
tail -f condor_logs/adp_$(Cluster)_$(Process).out

# 4. Debug a running job (optional)
condor_ssh_to_job <JOB_ID>
```


`adp.sub` unpacks `$(Process)` into (config index, seed) as follows:

```text
index = Process % N_CONFIGS
seed  = SEEDS[ Process / N_CONFIGS ]
```

### Output layout

The runner runs the trainers from `$OUTPUT_DIR`, so all output is relative to
it:

```text
$OUTPUT_DIR/
├── <method>/runs/<model_short>/<experiment_name>/   # per-run outputs & logs
│   ├── <experiment_name>_experiment_summary.txt     # val/test macro-F1, per-class metrics
│   ├── <experiment_name>_training_summary.txt       # per-epoch log, best vs final val F1
│   ├── <experiment_name>_experiment_summary.json
│   └── model_config.json
├── <method>/best_models/<model_short>/              # best-val checkpoint (archived)
├── experiments/                                     # registry (model_registry.json)
└── run_manifest.jsonl                               # (config, seed) skip-guard
```

`<experiment_name>` looks like `20260828_170858_p1_lr5e05_tbs8_ebs16_gas8`.

The skip-guard: every launched (config, seed) pair is recorded in
`run_manifest.jsonl`; pairs already marked `ok` are skipped on rerun, so an
interrupted batch can be resubmitted safely.

## 7. Key experiment settings

- Prompt templates & verbalizers are defined in `src/common_utils.py`
  (patterns 1–4).
- Fixed hyperparameters across the example configs: `max_length=512`,
  `warmup_ratio=0.1`, `weight_decay=0.01`, `class_weights=true`,
  `early_stopping_patience=5`, per-epoch evaluation, AdamW + cosine schedule,
  macro-F1 early stopping on the validation set.
- Early stopping: prompt — on (patience 5, unconditional); sequence — off
  (full epoch budget).
- Evaluation: the final in-memory (last) weights are evaluated on val/test;
  the best-val checkpoint is archived but never reloaded.
- Determinism: `set_random_seed` in `src/common_utils.py` seeds torch/numpy/
  random and enables deterministic cuDNN algorithms.

## 8. Collecting results

Each run's headline numbers are in `<experiment_name>_experiment_summary.txt`;
the per-epoch training log with best vs final validation F1 is in
`<experiment_name>_training_summary.txt`. For a final number, report the
mean ± std across seeds for each config.
