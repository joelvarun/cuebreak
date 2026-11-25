"""Draw docs/before_after.png from the numbers bench.py measured.

Run bench.py first, or pass --bench to run it here:
    python bench.py && python make_chart.py
    python make_chart.py --bench
"""

import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "docs", "results.json")
OUTPUT = os.path.join(HERE, "docs", "before_after.png")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
NEUTRAL = "#9d9c97"
NEUTRAL_LIGHT = "#c9c8c3"

#: The operating points quoted in the README. Both sit inside the caption
#: delay live broadcast subtitling already runs at, which is 3 to 6 seconds.
OPERATING_POINT = "cuebreak 4000ms"
BEST_EFFORT = "cuebreak 8000ms"
BASELINE = "greedy fill"
BASELINE_CPS = "greedy + cps"
OPTIMUM = "offline optimum"

#: The optimum is drawn as an outlined bar, not a filled one, because it is
#: a lower bound rather than something anyone can deploy: it needs the whole
#: stream before it can emit the first cue.
BAR_ORDER = [
    (BASELINE, "greedy fill\n(baseline)", NEUTRAL),
    (BASELINE_CPS, "greedy + cps\nflush", NEUTRAL_LIGHT),
    (OPERATING_POINT, "cuebreak\n4 s buffer", BLUE),
    (BEST_EFFORT, "cuebreak\n8 s buffer", AQUA),
    (OPTIMUM, "offline optimum\n(lower bound)", None),
]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8.5, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bars(ax, data, key, title, unit, fmt="%.1f"):
    labels, values, colours = [], [], []
    for name, label, colour in BAR_ORDER:
        labels.append(label)
        values.append(data["overall"][name][key])
        colours.append(colour)

    positions = list(range(len(values)))
    for x, value, colour in zip(positions, values, colours):
        if colour is None:
            ax.bar([x], [value], width=0.62, facecolor=SURFACE,
                   edgecolor=INK_SOFT, linewidth=1.3, hatch="///", zorder=3)
        else:
            ax.bar([x], [value], width=0.62, color=colour, zorder=3)

    headroom = (max(values) or 1.0) * 1.24
    for x, value in zip(positions, values):
        ax.text(x, value + headroom * 0.025, fmt % value, ha="center",
                va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, headroom)
    ax.set_ylabel(unit, color=INK_SOFT, fontsize=8.5)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)


def by_rate(ax, data):
    buckets = list(data["corpus"]["buckets"].items())
    series = [(BASELINE, "greedy fill", NEUTRAL),
              (OPERATING_POINT, "cuebreak, 4 s buffer", BLUE),
              (BEST_EFFORT, "cuebreak, 8 s buffer", AQUA)]
    width = 0.26
    for index, (name, label, colour) in enumerate(series):
        values = [data["by_rate"][name][bucket]["dangling_pct"]
                  for bucket, _ in buckets]
        offsets = [i + (index - 1) * width for i in range(len(buckets))]
        ax.bar(offsets, values, width=width - 0.025, color=colour,
               label=label, zorder=3)
        for x, value in zip(offsets, values):
            ax.text(x, value + 0.5, "%.0f" % value, ha="center", va="bottom",
                    fontsize=8, color=INK)
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(["%s\n%.0f wpm" % (name, info["words_per_minute"])
                        for name, info in buckets])
    ax.set_ylabel("percent of breaks", color=INK_SOFT, fontsize=8.5)
    ax.set_title("Breaks that strand a function word, by how fast the speaker "
                 "talks", color=INK, fontsize=10.5, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SOFT, loc="upper left")


def tradeoff(ax, data):
    budgets = data["lookaheads_ms"]
    points = [(data["overall"]["cuebreak %dms" % ms]["mean_word_delay_ms"],
               data["overall"]["cuebreak %dms" % ms]["cost_per_cue"], ms)
              for ms in budgets]
    ax.plot([p[0] for p in points], [p[1] for p in points], color=BLUE,
            linewidth=2.0, marker="o", markersize=6, zorder=4,
            label="cuebreak, audio buffer swept")
    for delay, cost, ms in points:
        if ms in (budgets[0], 4000, 8000):
            ax.annotate("%.1f s buffer" % (ms / 1000.0), (delay, cost),
                        textcoords="offset points", xytext=(6, 7),
                        fontsize=8, color=INK_SOFT)

    for name, label, colour, marker in (
            (BASELINE, "greedy fill", NEUTRAL, "s"),
            (BASELINE_CPS, "greedy + cps flush", "#6f6e69", "D")):
        row = data["overall"][name]
        ax.scatter([row["mean_word_delay_ms"]], [row["cost_per_cue"]],
                   color=colour, s=70, marker=marker, zorder=5, label=label)

    optimum = data["overall"][OPTIMUM]
    ax.axhline(optimum["cost_per_cue"], color=AQUA, linewidth=1.6,
               linestyle=(0, (4, 3)), zorder=3,
               label="offline optimum (needs the whole stream)")
    ax.set_xlabel("mean delay before a word reaches the screen (ms)",
                  color=INK_SOFT, fontsize=8.5)
    ax.set_ylabel("style penalty per cue", color=INK_SOFT, fontsize=8.5)
    ax.set_title("What caption delay buys: quality against latency",
                 color=INK, fontsize=10.5, loc="left", pad=8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%.0f" % v))
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SOFT,
              loc="upper right")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action="store_true",
                        help="run bench.py first")
    args = parser.parse_args()
    if args.bench or not os.path.exists(RESULTS):
        subprocess.check_call([sys.executable,
                               os.path.join(HERE, "bench.py")])
    with open(RESULTS, encoding="utf-8") as fh:
        data = json.load(fh)

    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), facecolor=SURFACE)
    for ax in axes.flat:
        style(ax)

    bars(axes[0][0], data, "dangling_pct",
         "Cue breaks that strand a function word", "percent of breaks",
         fmt="%.1f")
    bars(axes[0][1], data, "mean_cps_excess",
         "How far cues run over the 17 cps reading-speed ceiling",
         "characters per second over", fmt="%.2f")
    by_rate(axes[1][0], data)
    tradeoff(axes[1][1], data)

    corpus = data["corpus"]
    figure.suptitle("cuebreak: live caption segmentation on %d streams of "
                    "word-aligned meeting speech" % corpus["streams"],
                    color=INK, fontsize=13.5, x=0.012, ha="left", y=0.985)
    figure.text(0.012, 0.952,
                "AMI Meeting Corpus, %d words, %.0f minutes, %.0f words per "
                "minute mean. Caption style: %d characters, %d lines, %.0f "
                "cps ceiling. Lower is better in every panel."
                % (corpus["words"], corpus["seconds"] / 60.0,
                   corpus["words_per_minute"], data["model"]["max_chars"],
                   data["model"]["max_lines"], data["model"]["cps_limit"]),
                color=INK_SOFT, fontsize=9.5, ha="left")

    figure.tight_layout(rect=(0, 0, 1, 0.935))
    figure.savefig(OUTPUT, dpi=150, facecolor=SURFACE)
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
