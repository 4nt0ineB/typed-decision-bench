"""The reliability diagram, over every graded Jev answer this project has archived.

Day 1 called this the headline experiment: when Jev says 0.9, is it right 90% of the time
on my data? Nobody outside the company has published it. Every number here comes from
`results/probes/*.jsonl`, so this costs zero API calls and re-runs after any probe.

Two quantities get plotted separately, and keeping them apart is the point:

  p_chosen    the probability the model put on the answer it gave. THIS is what the
              calibration claim is about, and what ECE should be read on.
  confidence  a separate, undocumented number (session 6: not max-p, not normalised
              entropy, not Gini, not top-two margin). The documented patterns gate on it,
              so whether it is calibrated *as a probability* is a product question.

This chart checks the vendor's calibration claim and nothing else. It ranks nothing:
distance from the diagonal is honesty, not quality. Model comparisons, like-for-like on the
same items, are bench/report.py's job.

One honest caveat that the count panel exists to make visible: P5 and P6 are saturated
retrieval probes that scored 100% at p=1.00, so they pile ~2,700 answers into the top bin
and flatter any pooled average. Read the per-probe table, not just the curve.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict

from _common import RESULTS, ROOT

BINS = 10

# Newest run per probe, matching how P12 replays the P2 baseline. Earlier runs of the same
# probe are pilots at smaller n, and pooling them would weight a probe by how often it was
# re-run rather than by how many answers it graded.
JEV = {
    "P2 massive": ("p2-multilingual", "p_chosen", "confidence"),
    "P5 haystack": ("p5-haystack", "p_chosen", None),
    "P6 distractors": ("p6-distractors", "p_chosen", None),
    "P7 cross-record": ("p7-crossrecord", "p_chosen", "confidence"),
    "P8 prefill": ("p8-prefill", None, "certainty"),
    "P9 value pick": ("p9-value-selection", "p_chosen", "confidence"),
    "P11 free text": ("p11-freetext", None, "certainty"),
}

GROUPS = [(JEV, "#2a78d6", "jev-1.13.0")]


def read(prefix: str) -> list[dict]:
    runs = sorted(RESULTS.glob(f"{prefix}-*.jsonl"))
    if not runs:
        return []
    return [json.loads(line) for line in runs[-1].read_text(encoding="utf-8").splitlines()]


def series(spec: dict, key: str) -> dict[str, list[tuple[float, bool]]]:
    """probe -> [(probability, was_correct)], for whichever quantity `key` names."""
    out = {}
    for label, (prefix, p_field, conf_field) in spec.items():
        field = p_field if key == "p_chosen" else conf_field
        if not field:
            continue
        rows = [r for r in read(prefix)
                if r.get("correct") is not None and r.get(field) is not None]
        if rows:
            out[label] = [(float(r[field]), bool(r["correct"])) for r in rows]
    return out


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


def table(title: str, spec: dict, key: str) -> None:
    data = series(spec, key)
    if not data:
        return
    print(f"\n{title}")
    print(f"{'probe':<16} {'n':>6} {'accuracy':>9} {'mean p':>8} {'gap':>7} "
          f"{'ECE':>7} {'MCE':>7} {'Brier':>7}")
    for label, points in data.items():
        m = metrics(points)
        print(f"{label:<16} {m['n']:>6} {m['accuracy']:>8.1%} {m['mean_p']:>8.3f} "
              f"{m['mean_p'] - m['accuracy']:>+7.3f} {m['ece']:>7.3f} {m['mce']:>7.3f} "
              f"{m['brier']:>7.3f}")
    pooled = metrics([pt for points in data.values() for pt in points])
    print(f"{'POOLED':<16} {pooled['n']:>6} {pooled['accuracy']:>8.1%} {pooled['mean_p']:>8.3f} "
          f"{pooled['mean_p'] - pooled['accuracy']:>+7.3f} {pooled['ece']:>7.3f} "
          f"{pooled['mce']:>7.3f} {pooled['brier']:>7.3f}")


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

    figure, axes = plt.subplots(
        2, 2, figsize=(11, 7.4), sharex="col",
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12, "wspace": 0.22},
    )
    figure.patch.set_facecolor("#fcfcfb")

    for column, (key, heading) in enumerate([
        ("p_chosen", "Probability on the chosen answer"),
        ("confidence", "Reported confidence"),
    ]):
        curve, counts = axes[0][column], axes[1][column]
        for ax in (curve, counts):
            ax.set_facecolor("#fcfcfb")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=9)

        curve.plot([0, 1], [0, 1], color=GRID, linewidth=2, zorder=1)
        curve.text(0.99, 0.94, "perfectly calibrated", color=MUTED, fontsize=9, ha="right")

        for rank, (spec, colour, name) in enumerate(GROUPS):
            points = [pt for pts in series(spec, key).values() for pt in pts]
            if not points:
                continue
            bins = binned(points)
            curve.plot([b[0] for b in bins], [b[1] for b in bins], color=colour,
                       linewidth=2, marker="o", markersize=8, markeredgecolor="#fcfcfb",
                       markeredgewidth=2, label=f"{name}  (n={len(points):,})", zorder=3)
            # Bars side by side inside the 0.1-wide bin, not stacked on each other.
            width = 0.09 / len(GROUPS)
            offset = (rank - (len(GROUPS) - 1) / 2) * width
            counts.bar([b[0] + offset for b in bins], [b[2] for b in bins],
                       width=width, color=colour, linewidth=0)

        curve.set_title(heading, color=INK, fontsize=12, loc="left", pad=10)
        curve.set_ylabel("observed accuracy", color=MUTED, fontsize=10)
        curve.set_xlim(0, 1.02)
        curve.set_ylim(0, 1.02)
        curve.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
        curve.set_axisbelow(True)
        curve.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=INK)

        counts.set_yscale("log")
        counts.set_ylabel("answers", color=MUTED, fontsize=10)
        counts.set_xlabel(heading.lower(), color=MUTED, fontsize=10)
        counts.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
        counts.set_axisbelow(True)

    graded = sum(sum(r.get("correct") is not None for r in read(prefix))
                 for prefix, _, _ in JEV.values())
    figure.suptitle(
        f"Reliability on {graded:,} archived Jev answers",
        color=INK, fontsize=13.5, x=0.09, ha="left", y=0.98,
    )
    figure.text(0.09, 0.935,
                "Below the diagonal = overconfident. Pools every graded probe: read the "
                "per-probe table, not just the pooled curve.",
                color=MUTED, fontsize=9.5, ha="left")

    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    path = figures / "calibration.png"
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor=figure.get_facecolor())
    print(f"\nwrote {path}")


def main() -> None:
    table("jev-1.13.0, probability on the chosen answer", JEV, "p_chosen")
    table("jev-1.13.0, reported confidence", JEV, "confidence")
    plot()


if __name__ == "__main__":
    main()
