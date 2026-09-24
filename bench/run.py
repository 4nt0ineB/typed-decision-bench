"""Run one model on one task: results/bench/<model>/<task>.jsonl plus a <task>.json manifest.

    .venv/bin/python bench/run.py --model qwen3.5-4b --task massive_scenario --sample 100

Overwrites that model's previous run of that task. Every condition sees the same items.
`--split dev` writes <task>.dev.jsonl instead, fitting data that bench/report.py ignores.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.models import REGISTRY  # noqa: E402

BENCH = ROOT / "results" / "bench"
PACKAGES = ("typesafe-sdk", "laya", "torch", "transformers", "peft", "mlx", "mlx-lm")


def row(task: str, condition: str, item: str, state: str, gold: str, answer: dict,
        elapsed_ms: float) -> dict:
    probabilities = answer["probabilities"]
    p_chosen = probabilities[answer["predicted"]] if probabilities else None
    confidence = answer["confidence"]
    if confidence is None and probabilities:
        # P15: Jev's confidence is the chosen probability rescaled from [1/k, 1] to [0, 1].
        k = len(probabilities)
        confidence = (k * p_chosen - 1) / (k - 1)
    return {"task": task, "condition": condition, "id": item, "state": state, "gold": gold,
            "correct": answer["predicted"] == gold} | answer | {
            "p_chosen": p_chosen, "confidence": confidence, "elapsed_ms": elapsed_ms}


def run_file(task: str, split: str) -> str:
    return f"{task}.jsonl" if split == "test" else f"{task}.{split}.jsonl"


def write_manifest(path: Path, manifest: dict) -> None:
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")


def clock(seconds: float) -> str:
    return f"{int(seconds) // 60}m{int(seconds) % 60:02d}s"


def progress(task: str, done: int, total: int, started: float, last: str) -> None:
    elapsed = time.monotonic() - started
    eta = elapsed / done * (total - done)
    print(f"\r{task}  {done}/{total}  {done / total:.1%}  elapsed {clock(elapsed)}  "
          f"eta {clock(eta)}  last {last}\033[K", end="", file=sys.stderr, flush=True)


def installed(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def gpu() -> dict:
    # sys.modules, not an import: only a model that loaded torch ran on its GPU.
    torch = sys.modules.get("torch")
    return {"gpu": torch.cuda.get_device_name()} if torch and torch.cuda.is_available() else {}


def git_sha() -> str:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return git("rev-parse", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=REGISTRY)
    parser.add_argument("--task", required=True)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--split", default="test", choices=("test", "dev"))
    args = parser.parse_args()

    adapter, kwargs = REGISTRY[args.model]
    module = importlib.import_module(f"bench.models.{adapter}")
    if args.workers > 1 and not getattr(module, "CONCURRENT", False):
        raise SystemExit(f"--workers {args.workers}: adapter {adapter} is not CONCURRENT "
                         "(local model, one call at a time). Use --workers 1.")
    task = importlib.import_module(f"bench.tasks.{args.task}")
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    predict = module.load(**kwargs)
    pairs = task.load_pairs(args.sample, args.seed, args.split)
    work = [(condition, *task.question(condition), pair)
            for condition in task.CONDITIONS for pair in pairs]

    def call(item: tuple) -> dict:
        condition, lang, questions, pair = item
        start = time.perf_counter()
        answer = predict(pair[lang], questions)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return row(task.NAME, condition, pair["id"], pair[lang], pair["gold"], answer, elapsed_ms)

    def manifest(**status) -> dict:
        return {
            "model": args.model, "task": task.NAME, "adapter": adapter, "kwargs": kwargs,
            "split": args.split, "sample": args.sample, "seed": args.seed, "workers": args.workers, "n": len(rows),
        } | status | {
            "started": started, "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git": git_sha(), "platform": platform.platform(), "python": platform.python_version(),
            "versions": {p: installed(p) for p in PACKAGES},
        } | gpu() | getattr(predict, "manifest", {})

    path = BENCH / args.model / run_file(task.NAME, args.split)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    clock_start, shown = time.monotonic(), 0.0
    try:
        with path.open("w", encoding="utf-8") as jsonl, ThreadPoolExecutor(args.workers) as pool:
            # Sequential runs stay on the main thread: MLX and MPS state is per thread.
            for r in (pool.map if args.workers > 1 else map)(call, work):
                jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")
                jsonl.flush()
                rows.append(r)
                if time.monotonic() - shown >= 1 or len(rows) == len(work):
                    shown = time.monotonic()
                    progress(task.NAME, len(rows), len(work), clock_start,
                             f"{r['condition']} {r['id']}")
    except BaseException as exc:
        print(file=sys.stderr)
        write_manifest(path, manifest(complete=False, error=repr(exc)))
        raise
    print(file=sys.stderr)
    write_manifest(path, manifest(complete=True))

    by_condition = defaultdict(list)
    for r in rows:
        by_condition[r["condition"]].append(r["correct"])
    for condition, correct in by_condition.items():
        print(f"{condition:<8} {statistics.mean(correct):>6.1%}  n={len(correct)}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
