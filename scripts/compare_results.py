"""Compare scripts/eval.sh runs: clean / randomized / average per run (checkpoint and alpha selection), then
per task of the best run (or --detail): clean, randomized and the drop, sorted by the drop.

    python scripts/compare_results.py eval_result/output_full_*

Rates are over the tasks finished in every run, so partial runs and fixed subsets (TASKS=...) compare fairly.
"""
import argparse
import json
from pathlib import Path

SETTINGS = {"demo_clean": "clean", "demo_randomized": "random"}


def load(run):
    res = {name: {} for name in SETTINGS.values()}
    for tc, name in SETTINGS.items():
        for f in (run / tc / "eval_results").glob("*/result.json"):
            r = json.loads(f.read_text())
            res[name][f.parent.name] = (r["successes"], r["attempts"])
    return res


def rate(res, tasks):
    a = sum(res[t][1] for t in tasks)
    return 100 * sum(res[t][0] for t in tasks) / a if a else float("nan")


def main(a):
    runs = {Path(r).name: load(Path(r)) for r in a.runs}
    common = {name: set.intersection(*(set(d[name]) for d in runs.values())) for name in SETTINGS.values()}
    print(f"tasks finished in every run: clean {len(common['clean'])}, random {len(common['random'])}\n")
    table = []
    for name, d in runs.items():
        c, r = rate(d["clean"], common["clean"]), rate(d["random"], common["random"])
        table.append((name, c, r, (c + r) / 2))
    table.sort(key=lambda x: -x[3] if x[3] == x[3] else float("inf"))
    width = max(len(n) for n in runs) + 2
    print(f"{'run':<{width}}{'clean':>8}{'random':>8}{'avg':>8}")
    for name, c, r, avg in table:
        print(f"{name:<{width}}{c:>8.1f}{r:>8.1f}{avg:>8.1f}")

    detail = a.detail or table[0][0]
    d = runs[Path(detail).name]
    tasks = sorted(set(d["clean"]) | set(d["random"]))
    pct = lambda res, t: 100 * res[t][0] / res[t][1] if t in res and res[t][1] else float("nan")
    rows = [(t, pct(d["clean"], t), pct(d["random"], t)) for t in tasks]
    rows.sort(key=lambda x: -(x[1] - x[2]) if x[1] == x[1] and x[2] == x[2] else float("inf"))
    print(f"\nper task: {detail}  (random-only drop >= {a.drop:g}: augmentation; clean < {a.low:g}: training recipe)")
    print(f"{'task':<28}{'clean':>7}{'random':>8}{'drop':>7}")
    for t, c, r in rows:
        tag = "random drop" if c - r >= a.drop else "clean low" if c < a.low else ""
        print(f"{t:<28}{c:>7.0f}{r:>8.0f}{c - r:>7.0f}  {tag}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="eval_result/<run> dirs")
    p.add_argument("--detail", default=None, help="run for the per-task table (default: best average)")
    p.add_argument("--drop", type=float, default=30)
    p.add_argument("--low", type=float, default=50)
    main(p.parse_args())
