"""Weighted average of LingBot-VLA 2.0 checkpoints: WiSE-FT (fine-tuned <-> base) and model soups.

  B=/mnt/datadisk/models/lingbot-vla/lingbot-vla-v2-6b
  E=lingbot-vla-v2/output_full
  # WiSE-FT: 0.7 * fine-tuned + 0.3 * base
  python scripts/interpolate_weights.py $E/checkpoints/global_step_20000/hf_ckpt:0.7 $B:0.3 --out $E/checkpoints/wise0.7_20000/hf_ckpt
  # soup: equal weights
  python scripts/interpolate_weights.py $E/checkpoints/global_step_{15000,20000,25000}/hf_ckpt --out $E/checkpoints/soup_15-25k/hf_ckpt

Inputs are hf_ckpt or flat model dirs; weights must sum to 1 (none given: equal). Every tensor must exist in
every input with the same shape. Sums in fp32, saves in the first input's dtype and sharding; top-level
non-weight files (config, tokenizer) come from the first input. Put --out under <exp>/checkpoints/<name>/hf_ckpt
so that eval.sh finds <exp>/lingbotvla_cli.yaml.
--keys REGEX: only matching tensors are averaged, the rest are taken from the first input.
"""
import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INDEX = "model.safetensors.index.json"


def weight_map(d):
    if (d / INDEX).exists():
        return json.loads((d / INDEX).read_text())["weight_map"]
    with safe_open(d / "model.safetensors", "pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


def parse_input(s):
    path, _, w = s.rpartition(":")
    try:
        return Path(path).expanduser().resolve(), float(w)
    except ValueError:
        return Path(s).expanduser().resolve(), None


def main(a):
    inputs = [parse_input(s) for s in a.inputs]
    dirs = [d for d, _ in inputs]
    weights = [w for _, w in inputs]
    if all(w is None for w in weights):
        weights = [1 / len(dirs)] * len(dirs)
    elif any(w is None for w in weights) or abs(sum(weights) - 1) > 1e-6:
        raise SystemExit(f"give a weight to every input, summing to 1: {weights}")
    out = Path(a.out).expanduser().resolve()
    if out.exists():
        raise SystemExit(f"{out} exists")
    keys = re.compile(a.keys) if a.keys else None

    maps = [weight_map(d) for d in dirs]
    handles = {}

    def tensor_file(i, k):
        path = dirs[i] / maps[i][k]
        if path not in handles:
            handles[path] = safe_open(path, "pt")
        return handles[path]

    for d, m in zip(dirs[1:], maps[1:]):
        if set(m) != set(maps[0]):
            only0, only1 = sorted(set(maps[0]) - set(m)), sorted(set(m) - set(maps[0]))
            raise SystemExit(f"keys differ from {dirs[0]}: {d} lacks {only0[:5]} ({len(only0)}), has extra {only1[:5]} ({len(only1)})")
    for k in maps[0]:
        shapes = [tuple(tensor_file(i, k).get_slice(k).get_shape()) for i in range(len(dirs))]
        if len(set(shapes)) > 1:
            raise SystemExit(f"{k}: shapes differ {shapes}")

    for d, w in zip(dirs, weights):
        print(f"{w:.4f}  {d}")
    shards = defaultdict(list)
    for k, shard in maps[0].items():
        shards[shard].append(k)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    n_avg = n_copy = 0
    for shard, names in sorted(shards.items()):
        tensors = {}
        for k in names:
            first = tensor_file(0, k).get_tensor(k)
            if not first.is_floating_point() or (keys and not keys.search(k)):
                tensors[k] = first
                n_copy += 1
                continue
            acc = first.float() * weights[0]
            for i in range(1, len(dirs)):
                acc += tensor_file(i, k).get_tensor(k).float() * weights[i]
            tensors[k] = acc.to(first.dtype)
            n_avg += 1
        save_file(tensors, tmp / shard, metadata={"format": "pt"})
        print(f"{shard}: {len(names)} tensors", flush=True)
    for f in dirs[0].iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy2(f, tmp / f.name)
    (tmp / "interpolation.json").write_text(json.dumps(
        {"inputs": [str(d) for d in dirs], "weights": weights, "keys": a.keys}, indent=2))
    tmp.rename(out)
    print(f"averaged {n_avg} tensors, copied {n_copy} from the first input -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="model dir[:weight]")
    p.add_argument("--out", required=True)
    p.add_argument("--keys", default=None, help="regex: only average matching tensors")
    main(p.parse_args())
