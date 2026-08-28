#!/usr/bin/env python3
"""
Batch runner: drives src/train_prompt.py / src/train_sequence.py from the
JSONL config files (one line = one experiment). Each (config, seed) pair is
launched once; all outputs land under --output-dir (the trainers write their
timestamped experiments/ tree relative to their CWD, so the runner chdirs
there).

Two modes:
  run_all.sh mode:  python src/run_experiments.py --config C \
                        --device 0 --output-dir /scratch/$USER/adp_final
  adp.sub mode:     python src/run_experiments.py --config C --index I --seed S \
                        --output-dir /scratch/$USER/adp_final

Skip-guard: every launched pair is appended to <output-dir>/run_manifest.jsonl
with its return code. Pairs recorded as "ok" are skipped on rerun, so a
killed batch can simply be resubmitted. Failed pairs are retried.

Protocol notes:
  - prompt: early stopping ON (patience 5, unconditional in train_prompt.py)
  - sequence: no early stopping (train_sequence.py runs the full budget)
  - both: final in-memory model evaluated on val/test; 
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINERS = {
    "prompt": REPO_ROOT / "src" / "train_prompt.py",
    "sequence": REPO_ROOT / "src" / "train_sequence.py",
}


def build_cmd(entry, seed, data_path):
    """Map one JSONL config entry + seed to the trainer's CLI arguments."""
    method = entry["method"]
    if method not in TRAINERS:
        raise ValueError(f"unknown method {method!r} (expected prompt|sequence)")
    cmd = [
        sys.executable, str(TRAINERS[method]),
        "--model_id", str(entry["model_id"]),
        "--data_path", str(data_path),
        "--train_batch_size", str(entry.get("train_batch_size", 8)),
        "--eval_batch_size", str(entry.get("eval_batch_size", 16)),
        "--gradient_accumulation_steps", str(entry.get("gradient_accumulation_steps", 1)),
        "--num_epochs", str(entry.get("num_epochs", 15)),
        "--max_length", str(entry.get("max_length", 512)),
        "--lr", str(entry["lr"]),
        "--early_stopping_patience", str(entry.get("early_stopping_patience", 5)),
        "--warmup_ratio", str(entry.get("warmup_ratio", 0.1)),
        "--weight_decay", str(entry.get("weight_decay", 0.01)),
        "--seed", str(seed),
    ]
    if method == "prompt":
        cmd += ["--prompt_pattern", str(entry.get("prompt_pattern", 1))]
    if entry.get("class_weights", False):
        cmd.append("--class_weights")
    return cmd


def load_manifest(manifest_path):
    """Return the set of (config_index, seed) pairs already run successfully."""
    done = set()
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "ok":
                done.add((rec["config_index"], rec["seed"]))
    return done


def append_manifest(manifest_path, record):
    with manifest_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="JSONL config file")
    parser.add_argument("--seeds", default=None,
                        help="comma-separated seeds (sweep mode)")
    parser.add_argument("--seed", type=int, default=None,
                        help="single seed, overrides --seeds (adp.sub mode)")
    parser.add_argument("--index", type=int, default=None,
                        help="run only this config index (adp.sub mode)")
    parser.add_argument("--device", default=None,
                        help="GPU index for CUDA_VISIBLE_DEVICES (ignored if the "
                             "variable is already set, e.g. by HTCondor)")
    parser.add_argument("--output-dir", required=True,
                        help="where experiments/, logs and run_manifest.jsonl go")
    parser.add_argument("--data-path", default=None,
                        help="default: <repo root>/data")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the planned commands without running them")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.data_path or str(REPO_ROOT / "data")

    entries = [json.loads(line) for line in
               Path(args.config).read_text().splitlines() if line.strip()]
    indexes = [args.index] if args.index is not None else range(len(entries))
    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [42]

    env = dict(os.environ)
    if args.device is not None and "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = str(args.device)

    manifest_path = output_dir / "run_manifest.jsonl"
    done = load_manifest(manifest_path)

    n_ok = n_skip = n_fail = 0
    for idx in indexes:
        if idx < 0 or idx >= len(entries):
            print(f"!! config index {idx} out of range (0..{len(entries) - 1})",
                  file=sys.stderr)
            continue
        entry = entries[idx]
        method = entry["method"]
        tag = entry.get("experiment_id", f"cfg{idx}")
        for seed in seeds:
            if (idx, seed) in done:
                print(f"== skip {tag} (index {idx}) seed {seed}: already ok")
                n_skip += 1
                continue
            cmd = build_cmd(entry, seed, data_path)
            label = f"{tag} (index {idx}) seed {seed} [{method}]"
            print(f"\n========== [{time.strftime('%F %T')}] {label} ==========")
            print(" ", " ".join(cmd), flush=True)
            if args.dry_run:
                continue
            record = {
                "config_index": idx, "seed": seed, "experiment_id": tag,
                "method": method, "cmd": " ".join(cmd),
            }
            t0 = time.time()
            result = subprocess.run(cmd, cwd=str(output_dir), env=env)
            record["seconds"] = round(time.time() - t0, 1)
            record["status"] = "ok" if result.returncode == 0 else "failed"
            append_manifest(manifest_path, record)
            if result.returncode == 0:
                n_ok += 1
            else:
                n_fail += 1
                print(f"!!!! {label}: FAILED (rc={result.returncode}), "
                      f"continuing with the next pair !!!!", file=sys.stderr)

    print(f"\n=== sweep done: ok={n_ok} failed={n_fail} skipped={n_skip} "
          f"manifest={manifest_path} ===")


if __name__ == "__main__":
    main()
