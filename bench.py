"""Benchmark caption segmentation on word-aligned meeting speech.

Compares the two greedy rules that shipping caption writers use against the
bounded-lookahead segmenter at several right-context budgets, and against the
exact offline optimum, which is a lower bound no policy can beat.

Run:
    python bench.py
    python bench.py --out docs/results.json
"""

import argparse
import json
import os
import statistics
import time
from typing import Dict, List

from cuebreak import CueModel, load_streams, segment_greedy, segment_optimal
from cuebreak.baseline import segment_greedy_cps
from cuebreak.online import segment_online
from cuebreak.model import break_penalty
from cuebreak.optimal import Segmentation
from cuebreak.streams import Stream, rate_buckets

HERE = os.path.dirname(os.path.abspath(__file__))
LOOKAHEADS_MS = [500, 1000, 1500, 2000, 3000, 4000, 6000, 8000]

#: A break penalty this high means the cue ended on a determiner, preposition
#: or auxiliary, leaving the reader holding a phrase that has not arrived.
DANGLING = 3.0


def measure(stream: Stream, seg: Segmentation, model: CueModel) -> Dict:
    """Per-cue statistics for one segmentation of one stream."""
    words = stream.words
    n = len(words)
    over = dangling = two_line = 0
    excess: List[float] = []
    imbalance: List[int] = []
    delays: List[int] = []
    for index, cue in enumerate(seg.cues):
        if cue.cps > model.cps_limit:
            over += 1
        excess.append(max(0.0, cue.cps - model.cps_limit))
        if cue.end_index < n:
            if break_penalty(words[cue.end_index - 1].text,
                             words[cue.end_index].text) >= DANGLING:
                dangling += 1
        if len(cue.lines) == 2:
            two_line += 1
            imbalance.append(abs(len(cue.lines[0]) - len(cue.lines[1])))
        if seg.horizons:
            horizon = seg.horizons[index]
            shown_at = (words[horizon].start_ms if horizon < n
                        else stream.end_of_audio_ms)
            # What the viewer experiences: how long after each word was
            # spoken does the cue carrying it reach the screen.
            delays.extend(max(0, shown_at - words[k].start_ms)
                          for k in range(cue.start_index, cue.end_index))
    return {
        "cues": len(seg.cues),
        "cost": seg.cost,
        "over_cps": over,
        "dangling": dangling,
        "breaks": sum(1 for c in seg.cues if c.end_index < n),
        "two_line": two_line,
        "imbalance": imbalance,
        "excess": excess,
        "delays": delays,
        "seconds": stream.end_of_audio_ms / 1000.0,
    }


def aggregate(rows: List[Dict]) -> Dict:
    cues = sum(r["cues"] for r in rows)
    breaks = sum(r["breaks"] for r in rows)
    two_line = sum(r["two_line"] for r in rows)
    seconds = sum(r["seconds"] for r in rows)
    imbalance = [v for r in rows for v in r["imbalance"]]
    excess = [v for r in rows for v in r["excess"]]
    delays = [v for r in rows for v in r["delays"]]
    delays.sort()
    return {
        "streams": len(rows),
        "cues": cues,
        "cost_total": sum(r["cost"] for r in rows),
        "cost_per_cue": sum(r["cost"] for r in rows) / cues,
        "over_cps_pct": 100.0 * sum(r["over_cps"] for r in rows) / cues,
        "mean_cps_excess": statistics.mean(excess) if excess else 0.0,
        "dangling_pct": (100.0 * sum(r["dangling"] for r in rows) / breaks
                         if breaks else 0.0),
        "mean_imbalance": statistics.mean(imbalance) if imbalance else 0.0,
        "two_line_pct": 100.0 * two_line / cues,
        "cues_per_minute": cues / (seconds / 60.0),
        "mean_word_delay_ms": statistics.mean(delays) if delays else 0.0,
        "p95_word_delay_ms": (delays[int(0.95 * (len(delays) - 1))]
                              if delays else 0),
    }


def methods(model: CueModel):
    yield "greedy fill", lambda s: segment_greedy(s.words, model,
                                                  s.end_of_audio_ms)
    yield "greedy + cps", lambda s: segment_greedy_cps(s.words, model,
                                                       s.end_of_audio_ms)
    for budget in LOOKAHEADS_MS:
        yield ("cuebreak %dms" % budget,
               lambda s, ms=budget: segment_online(s.words, model,
                                                   s.end_of_audio_ms, ms))
    yield "offline optimum", lambda s: segment_optimal(s.words, model,
                                                       s.end_of_audio_ms)


def run(model: CueModel) -> Dict:
    streams = load_streams()
    buckets = rate_buckets(streams)
    results: Dict[str, Dict] = {}
    per_bucket: Dict[str, Dict[str, Dict]] = {}

    for name, fn in methods(model):
        started = time.perf_counter()
        rows = {id(s): measure(s, fn(s), model) for s in streams}
        elapsed = time.perf_counter() - started
        results[name] = aggregate(list(rows.values()))
        results[name]["seconds_to_run"] = elapsed
        per_bucket[name] = {
            label: aggregate([rows[id(s)] for s in group])
            for label, group in buckets
        }

    return {
        "model": {
            "max_chars": model.max_chars, "max_lines": model.max_lines,
            "cps_limit": model.cps_limit,
            "min_duration_ms": model.min_duration_ms,
            "max_duration_ms": model.max_duration_ms,
        },
        "corpus": {
            "streams": len(streams),
            "words": sum(len(s.words) for s in streams),
            "seconds": sum(s.end_of_audio_ms for s in streams) / 1000.0,
            "words_per_minute": statistics.mean(s.words_per_minute
                                                for s in streams),
            "buckets": {label: {
                "streams": len(group),
                "words_per_minute": statistics.mean(x.words_per_minute
                                                    for x in group),
                "chars_per_second": statistics.mean(x.chars_per_second
                                                    for x in group),
            } for label, group in buckets},
        },
        "lookaheads_ms": LOOKAHEADS_MS,
        "overall": results,
        "by_rate": per_bucket,
    }


def report(data: Dict) -> None:
    corpus = data["corpus"]
    print("corpus: %d streams, %d words, %.0f s of speech, %.0f wpm mean"
          % (corpus["streams"], corpus["words"], corpus["seconds"],
             corpus["words_per_minute"]))
    for label, info in corpus["buckets"].items():
        print("  %-7s %4d streams  %5.1f wpm  %5.2f cps"
              % (label, info["streams"], info["words_per_minute"],
                 info["chars_per_second"]))
    print()
    header = ("method", "cues", "over 17cps", "cps excess", "dangling",
              "imbalance", "cost/cue", "word delay")
    print("%-17s %6s %11s %11s %9s %10s %9s %11s" % header)
    print("-" * 94)
    for name, row in data["overall"].items():
        print("%-17s %6d %10.1f%% %11.2f %8.1f%% %10.1f %9.3f %7.0f ms"
              % (name, row["cues"], row["over_cps_pct"],
                 row["mean_cps_excess"], row["dangling_pct"],
                 row["mean_imbalance"], row["cost_per_cue"],
                 row["mean_word_delay_ms"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(HERE, "docs",
                                                      "results.json"))
    parser.add_argument("--cps-limit", type=float, default=17.0)
    parser.add_argument("--max-chars", type=int, default=42)
    args = parser.parse_args()

    model = CueModel(cps_limit=args.cps_limit, max_chars=args.max_chars)
    data = run(model)
    report(data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
