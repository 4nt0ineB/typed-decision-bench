"""Regenerate the bench table and charts from results/bench/*/*.jsonl. Zero model calls.

Per task: results/bench/summary.md (also printed), figures/risk-coverage-<task>.png and
figures/reliability-<task>.png, one row and one curve per model directory.

The risk-coverage chart is the one that ranks: gate on confidence, keep the head, and read
off how much traffic each model handles at what accuracy. The reliability chart checks
calibration and ranks nothing: a model can hug the diagonal while being useless.

Rows without probabilities (generate-only adapters) count for accuracy and latency, not for
ECE, Brier or coverage.
"""

from __future__ import annotations

import importlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.models import CHARTED, PRICES, REGISTRY  # noqa: E402
from bench.run import BENCH  # noqa: E402

# The thresholds the vendor's own patterns gate on.
GATES = [0.60, 0.85, 0.90]
BINS = 10
INK, MUTED, GRID, PAPER = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb"


def curve(points: list[tuple[float, bool]], floor: float = 0.02):
    """Coverage and accuracy-on-the-retained-head, one point per distinct confidence.

    Every threshold keeps all its tied answers: walking rows one by one would order ties
    arbitrarily and draw accuracy the gate cannot deliver.

    Clipped below `floor`: the first handful of answers gives a denominator of two or
    three, so the left edge is a spike of pure sampling noise that reads as a real
    difference between models.
    """
    ordered = sorted(points, key=lambda point: point[0], reverse=True)
    coverage, accuracy, right = [], [], 0
    for kept, (confidence, correct) in enumerate(ordered, start=1):
        right += correct
        if kept < len(ordered) and ordered[kept][0] == confidence:
            continue
        if kept / len(ordered) < floor:
            continue
        coverage.append(kept / len(ordered))
        accuracy.append(right / kept)
    return coverage, accuracy


def at_gate(points: list[tuple[float, bool]], gate: float) -> tuple[float, float] | None:
    kept = [correct for conf, correct in points if conf >= gate]
    if not kept:
        return None
    return len(kept) / len(points), sum(kept) / len(kept)


def automatable(points: list[tuple[float, bool]], budget: float = 0.05) -> float:
    """The largest share one threshold passes with at most `budget` of it wrong. Each model
    gets its own threshold, so one that ships flatter probabilities is not penalised."""
    return max((c for c, a in zip(*curve(points, floor=0)) if a >= 1 - budget), default=0.0)


def binned(points: list[tuple[float, bool]]) -> list[tuple[float, float, int]]:
    buckets = defaultdict(list)
    for p, correct in points:
        buckets[min(int(p * BINS), BINS - 1)].append((p, correct))
    return [
        (statistics.mean(p for p, _ in vals),
         statistics.mean(c for _, c in vals),
         len(vals))
        for _, vals in sorted(buckets.items())
    ]


def metrics(points: list[tuple[float, bool]]) -> dict[str, float]:
    n = len(points)
    bins = binned(points)
    return {
        "n": n,
        "accuracy": statistics.mean(c for _, c in points),
        "mean_p": statistics.mean(p for p, _ in points),
        "ece": sum(count / n * abs(mean_p - acc) for mean_p, acc, count in bins),
        "mce": max(abs(mean_p - acc) for mean_p, acc, _ in bins),
        "brier": statistics.mean((p - c) ** 2 for p, c in points),
    }


def wilson(right: int, n: int, z: float = 1.96) -> tuple[float, float]:
    centre = (right + z * z / 2) / (n + z * z)
    half = z * math.sqrt(right * (n - right) / n + z * z / 4) / (n + z * z)
    return centre - half, centre + half


def discover() -> dict[str, dict[str, list[dict]]]:
    """task -> model -> rows."""
    tasks: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for path in sorted(BENCH.glob("*/*.jsonl")):
        if "." in path.stem:  # <task>.dev.jsonl: fitting data, never scored
            continue
        tasks[path.stem][path.parent.name] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return tasks


def source(model: str, task: str) -> str:
    manifest = json.loads((BENCH / model / f"{task}.json").read_text(encoding="utf-8"))
    if manifest.get("derived"):
        return f"derived from {Path(manifest['source']).parent.name}"
    return f"run {manifest['started']}"


def cost(model: str, rows: list[dict]) -> str:
    if model not in PRICES:
        return "local"
    per_in, per_out = PRICES[model]
    return f"{statistics.mean(r['input_tokens'] * per_in + r['output_tokens'] * per_out for r in rows) / 1e3:.4f}"


def scored(rows: list[dict], key: str) -> list[tuple[float, bool]]:
    return [(r[key], r["correct"]) for r in rows if r["probabilities"] is not None]


def table(task: str, models: dict[str, list[dict]]) -> str:
    module = importlib.import_module(f"bench.tasks.{task}")
    conditions = list(module.CONDITIONS)
    header = (["model", "n", "acc", "95% CI", "ECE", "Brier", "cover@0.90", "acc@0.90", "auto@5%"]
              + [f"{c} acc / ECE" for c in conditions] + ["p50 ms", "in tok", "$/1k items", "source"])
    lines = [f"## {task}", "", "| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for model, rows in models.items():
        right = sum(r["correct"] for r in rows)
        low, high = wilson(right, len(rows))
        points = scored(rows, "p_chosen")
        m = metrics(points) if points else None
        gate = at_gate(scored(rows, "confidence"), 0.90) if points else None
        cells = [model, str(len(rows)), f"{right / len(rows):.1%}", f"{low:.1%}-{high:.1%}",
                 f"{m['ece']:.3f}" if m else "-", f"{m['brier']:.3f}" if m else "-",
                 f"{gate[0]:.1%}" if gate else "-", f"{gate[1]:.1%}" if gate else "-",
                 f"{automatable(scored(rows, 'confidence')):.1%}" if points else "-"]
        for condition in conditions:
            subset = [r for r in rows if r["condition"] == condition]
            sub_points = scored(subset, "p_chosen")
            cells.append(
                (f"{statistics.mean(r['correct'] for r in subset):.0%}" if subset else "-")
                + (f" / {metrics(sub_points)['ece']:.3f}" if sub_points else ""))
        cells += [f"{statistics.median(r['elapsed_ms'] for r in rows):.0f}",
                  f"{statistics.mean(r['input_tokens'] for r in rows):.0f}",
                  cost(model, rows), source(model, task)]
        lines.append("| " + " | ".join(cells) + " |")
    # A task can add its own section under the shared table, e.g. records_count's error sizes.
    extra = getattr(module, "report", None)
    return ("\n".join(lines) + "\n\n" + gated(task, conditions, models)
            + (f"\n{extra(models)}" if extra else ""))


def gated(task: str, conditions: list[str], models: dict[str, list[dict]]) -> str:
    header = ["model"] + [f"{c} {k}" for c in conditions for k in ("at 0.90", "automatable")]
    lines = [f"### {task}: what a threshold lets through, per condition", "",
             "*At 0.90*: share of answers whose confidence clears 0.90 / how often those are "
             "right. *Automatable*: the largest share one threshold, set for that model, passes "
             "with at most 5% of it wrong.", "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for model, rows in models.items():
        cells = [model]
        for condition in conditions:
            points = scored([r for r in rows if r["condition"] == condition], "confidence")
            gate = at_gate(points, 0.90) if points else None
            cells += [f"{gate[0]:.0%} / {gate[1]:.0%}" if gate else "-",
                      f"{automatable(points):.0%}" if points else "-"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def style(ax) -> None:
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def risk_coverage(plt, task: str, models: dict[str, list[dict]]) -> Path:
    from matplotlib.lines import Line2D

    figure, ax = plt.subplots(figsize=(9.5, 6), facecolor=PAPER)
    style(ax)
    handles = []
    # Drawn last-listed first, so the first model's curve is never hidden under another.
    for rank, (model, rows) in reversed(list(enumerate(models.items()))):
        points = scored(rows, "confidence")
        if not points:
            continue
        label, colour, marker = CHARTED[model]
        x, y = curve(points)
        ax.plot(x, y, color=colour, linewidth=2.2, zorder=3 + 1 / (rank + 1))
        for gate in GATES:
            spot = at_gate(points, gate)
            if spot:
                ax.plot(*spot, marker=marker, markersize=10, color=colour,
                        markeredgecolor=PAPER, markeredgewidth=1.5, zorder=5 + 1 / (rank + 1))
        handles.insert(0, Line2D([], [], color=colour, linewidth=2.2, marker=marker,
                                 markersize=9, markeredgecolor=PAPER, label=label))

    ax.set_xlim(0, 1.02)
    ax.set_ylim(top=1.01)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("share of traffic handled automatically", color=MUTED, fontsize=10.5)
    ax.set_ylabel("accuracy on what was handled", color=MUTED, fontsize=10.5)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False,
              fontsize=10, labelcolor=INK)
    figure.suptitle("How much can you automate, and how accurate is what you automate?",
                    color=INK, fontsize=14, x=0.065, ha="left", y=0.99)
    figure.text(0.065, 0.905,
                f"{task}. You automate the answers whose confidence clears a threshold and send the rest "
                "to a human. Lowering the threshold moves right.\n"
                "Higher is better. Dots, left to right: threshold at 0.90, 0.85 and 0.60.",
                color=MUTED, fontsize=9.5, ha="left")
    return save(plt, figure, f"risk-coverage-{task}.png")


def reliability(plt, task: str, models: dict[str, list[dict]]) -> Path:
    figure, (top, counts) = plt.subplots(
        2, 1, figsize=(7, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
    figure.patch.set_facecolor(PAPER)
    for ax in (top, counts):
        style(ax)

    top.plot([0, 1], [0, 1], color=GRID, linewidth=2, zorder=1, label="perfectly calibrated")
    plotted = {m: scored(rows, "p_chosen") for m, rows in models.items()}
    plotted = {m: points for m, points in plotted.items() if points}
    # Bars side by side inside the 0.1-wide bin, not stacked on each other.
    width = 0.09 / len(plotted)
    for rank, (model, points) in enumerate(plotted.items()):
        bins = binned(points)
        label, colour, marker = CHARTED[model]
        top.plot([b[0] for b in bins], [b[1] for b in bins], color=colour,
                 linewidth=2, marker=marker, markersize=9, markeredgecolor=PAPER,
                 markeredgewidth=1.5, label=f"{label}  (n={len(points):,})", zorder=3)
        offset = (rank - (len(plotted) - 1) / 2) * width
        counts.bar([b[0] + offset for b in bins], [b[2] for b in bins],
                   width=width, color=colour, linewidth=0)

    top.set_ylabel("how often it was actually right", color=MUTED, fontsize=10)
    top.set_xlim(0, 1.05)
    top.set_ylim(0, 1.02)
    top.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    top.set_axisbelow(True)
    top.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9.5,
               labelcolor=INK)
    counts.set_yscale("log")
    counts.set_ylabel("answers", color=MUTED, fontsize=10)
    counts.set_xlabel("confidence the model gave its answer", color=MUTED, fontsize=10)
    counts.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    counts.set_axisbelow(True)
    figure.suptitle("When a model says 80% sure, is it right 80% of the time?", color=INK,
                    fontsize=13.5, x=0.09, ha="left", y=0.99)
    figure.text(0.09, 0.925, f"{task}. On the diagonal = honest. Below = overconfident, "
                "above = underconfident.\nBottom panel: how many answers at each "
                "confidence level.", color=MUTED, fontsize=9.5, ha="left")
    return save(plt, figure, f"reliability-{task}.png")


def save(plt, figure, name: str) -> Path:
    path = ROOT / "figures" / name
    path.parent.mkdir(exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return path


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = discover()
    found = {m for models in tasks.values() for m in models}
    order = list(REGISTRY) + sorted(found - set(REGISTRY))

    sections = []
    for task, models in sorted(tasks.items()):
        models = {m: models[m] for m in order if m in models}
        sections.append(table(task, models))
        print(sections[-1])
        charted = {m: models[m] for m in CHARTED if m in models}
        print(f"wrote {risk_coverage(plt, task, charted)}")
        print(f"wrote {reliability(plt, task, charted)}\n")
    summary = BENCH / "summary.md"
    summary.write_text("# Bench summary\n\nGenerated by bench/report.py from results/bench/.\n\n"
                       + "\n".join(sections), encoding="utf-8")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
