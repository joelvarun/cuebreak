"""Tests for the cue model, the exact segmenter and the online segmenter.

Run with pytest, or directly:
    python -m pytest tests -q
    python tests/test_cuebreak.py
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cuebreak import CueModel, load_streams, segment_greedy, segment_optimal
from cuebreak.baseline import segment_greedy_cps, wrap_first_fit
from cuebreak.online import OnlineSegmenter
from cuebreak.optimal import suffix_lower_bounds
from cuebreak.vtt import to_webvtt

MODEL = CueModel()
STREAMS = load_streams()
SAMPLE = STREAMS[:150]


def brute_force(words, model, end_of_audio_ms):
    """Minimum cost over every segmentation, by enumeration."""
    n = len(words)
    best = float("inf")
    for size in range(n):
        for cuts in itertools.combinations(range(1, n), size):
            bounds = list(cuts) + [n]
            total, start, ok = 0.0, 0, True
            for end in bounds:
                cue = model.cue_cost(words, start, end, end_of_audio_ms)
                if cue is None:
                    ok = False
                    break
                total += cue.cost
                start = end
            if ok:
                best = min(best, total)
    return best


def test_dp_matches_brute_force():
    """The forward pass really is the minimum, not a good heuristic."""
    small = [s for s in STREAMS if 5 <= len(s.words) <= 11][:12]
    assert small, "no short streams to enumerate"
    for stream in small:
        exact = segment_optimal(stream.words, MODEL, stream.end_of_audio_ms)
        reference = brute_force(stream.words, MODEL, stream.end_of_audio_ms)
        assert abs(exact.cost - reference) < 1e-9, (
            "%s: dp %.6f, enumeration %.6f"
            % (stream.text[:40], exact.cost, reference))


def test_optimum_is_a_lower_bound_for_every_policy():
    """No causal or offline policy in this repository can beat the optimum."""
    for stream in SAMPLE:
        best = segment_optimal(stream.words, MODEL, stream.end_of_audio_ms)
        others = [segment_greedy(stream.words, MODEL, stream.end_of_audio_ms),
                  segment_greedy_cps(stream.words, MODEL,
                                     stream.end_of_audio_ms)]
        others += [OnlineSegmenter(MODEL, ms).run(stream.words,
                                                  stream.end_of_audio_ms)
                   for ms in (500, 1000, 2000, 4000, 8000, 60000)]
        for other in others:
            assert other.cost >= best.cost - 1e-9


def test_suffix_bounds_agree_with_forward_pass():
    for stream in SAMPLE[:40]:
        forward = segment_optimal(stream.words, MODEL, stream.end_of_audio_ms)
        tail = suffix_lower_bounds(stream.words, MODEL, stream.end_of_audio_ms)
        assert abs(tail[0] - forward.cost) < 1e-9


def test_stream_sized_buffer_reaches_the_optimum():
    """With the whole stream buffered the online pass is the offline pass."""
    for stream in SAMPLE:
        best = segment_optimal(stream.words, MODEL, stream.end_of_audio_ms)
        full = OnlineSegmenter(MODEL, stream.end_of_audio_ms + 1000).run(
            stream.words, stream.end_of_audio_ms)
        # Only the cost has to match. Distinct segmentations can tie, and
        # the window pass and the forward pass break ties differently.
        assert abs(full.cost - best.cost) < 1e-9, (
            "%.6f vs %.6f" % (full.cost, best.cost))


def test_every_cue_fits_the_caption_area():
    for stream in SAMPLE:
        runs = {
            "optimal": segment_optimal(stream.words, MODEL,
                                       stream.end_of_audio_ms).cues,
            "greedy": segment_greedy(stream.words, MODEL,
                                     stream.end_of_audio_ms).cues,
            "greedy_cps": segment_greedy_cps(stream.words, MODEL,
                                             stream.end_of_audio_ms).cues,
            "online": OnlineSegmenter(MODEL, 2000).run(
                stream.words, stream.end_of_audio_ms).cues,
        }
        for name, cues in runs.items():
            for cue in cues:
                assert len(cue.lines) <= MODEL.max_lines, name
                assert max(len(ln) for ln in cue.lines) <= MODEL.max_chars, name


def test_segmentations_cover_every_word_once():
    for stream in SAMPLE:
        n = len(stream.words)
        for cues in (segment_optimal(stream.words, MODEL,
                                     stream.end_of_audio_ms).cues,
                     segment_greedy(stream.words, MODEL,
                                    stream.end_of_audio_ms).cues,
                     OnlineSegmenter(MODEL, 3000).run(
                         stream.words, stream.end_of_audio_ms).cues):
            assert cues[0].start_index == 0
            assert cues[-1].end_index == n
            for left, right in zip(cues, cues[1:]):
                assert left.end_index == right.start_index


def test_first_fit_wrap_never_loses_a_word():
    for stream in SAMPLE[:40]:
        end = min(MODEL.max_end(stream.words, 0), len(stream.words))
        lines = wrap_first_fit(stream.words, 0, end, MODEL)
        assert " ".join(lines).split() == [w.text for w in
                                           stream.words[0:end]]


def test_bigger_buffer_delays_the_first_decision():
    """A larger buffer cannot be ready sooner, decision by decision.

    Only the first decision can be compared directly: once the committed
    boundary moves, later cues start from different words, so their individual
    delays are not comparable. The corpus mean is checked separately.
    """
    for stream in SAMPLE[:60]:
        taken = []
        for ms in (500, 1000, 2000, 4000, 8000):
            first = OnlineSegmenter(MODEL, ms).run(
                stream.words, stream.end_of_audio_ms).decisions[0]
            taken.append(first.decided_at_ms)
        assert taken == sorted(taken), taken


def test_bigger_buffer_costs_delay_on_average():
    """Across the corpus, buying context costs caption lag."""
    means = []
    for ms in (500, 1000, 2000, 4000):
        delays = []
        for stream in SAMPLE[:60]:
            delays.extend(OnlineSegmenter(MODEL, ms).run(
                stream.words, stream.end_of_audio_ms).delays())
        means.append(sum(delays) / float(len(delays)))
    assert means == sorted(means), means


def test_webvtt_timestamps_are_ordered():
    stream = SAMPLE[3]
    cues = segment_optimal(stream.words, MODEL, stream.end_of_audio_ms).cues
    text = to_webvtt(cues)
    assert text.startswith("WEBVTT")
    for cue in cues:
        assert cue.start_ms < cue.end_ms
    for left, right in zip(cues, cues[1:]):
        assert left.end_ms <= right.start_ms + 1


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("ok   %s" % name)
        except AssertionError as exc:
            failures += 1
            print("FAIL %s: %s" % (name, exc))
    print("%d failure(s)" % failures)
    sys.exit(1 if failures else 0)
