"""Greedy baselines: what shipping live-caption writers actually do.

`segment_greedy` is fill-to-width with a sentence flush. It is the rule behind
whisper.cpp's --max-len splitter, the SubRip and WebVTT writers in common
media tooling, and most streaming-recogniser caption bridges: append words
until the next one will not fit in the caption area, until the cue has been up
for its maximum duration, or until a sentence ends, then flush. Lines are
wrapped first fit, filling the top line before starting the second.

`segment_greedy_cps` is the hardening a practitioner reaches for once viewers
complain that captions flash past: the same rule, but it closes the cue at the
largest boundary that keeps the cue inside the reading-speed ceiling. It sees
the same candidate boundaries the bounded-lookahead segmenter sees at zero
right context, so the comparison is not rigged by information. What it cannot
do is weigh reading speed against where the break lands, because it commits
one cue at a time with no view of the cue after it.
"""

from typing import List, Optional, Tuple

from .model import CLOSERS, STRONG_PUNCT, Cue, CueModel, Word
from .optimal import Segmentation


def wrap_first_fit(words: List[Word], start: int, end: int,
                   model: CueModel) -> List[str]:
    """Fill the top line to the character ceiling, then start the next."""
    lines: List[str] = []
    current = ""
    for word in words[start:end]:
        candidate = word.text if not current else current + " " + word.text
        if len(candidate) <= model.max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word.text
    if current:
        lines.append(current)
    return lines


def _fits(words: List[Word], start: int, end: int, model: CueModel) -> bool:
    """True when the first-fit wrap stays inside the caption area."""
    lines = wrap_first_fit(words, start, end, model)
    return len(lines) <= model.max_lines and all(
        len(line) <= model.max_chars for line in lines)


def _emit(words: List[Word], start: int, end: int, model: CueModel,
          end_of_audio_ms: int) -> Cue:
    """Score the cue a greedy writer produces, with the writer's own wrap."""
    lines = wrap_first_fit(words, start, end, model)
    cue = model.cue_cost(words, start, end, end_of_audio_ms, lines=lines)
    if cue is None:
        raise ValueError("greedy produced an unrenderable cue")
    return cue


def _ends_sentence(word: Word) -> bool:
    return word.text.rstrip(CLOSERS).endswith(STRONG_PUNCT)


def _candidates(words: List[Word], start: int, model: CueModel) -> List[int]:
    """Closing boundaries that render inside the caption area."""
    limit = min(model.max_end(words, start), len(words))
    return [j for j in range(start + 1, limit + 1)
            if _fits(words, start, j, model)] or [start + 1]


def _flush_point(words: List[Word], start: int, model: CueModel,
                 end_of_audio_ms: int, cps_aware: bool) -> Tuple[int, int]:
    """Return (closing boundary, last word index the rule had to observe)."""
    options = _candidates(words, start, model)
    widest = options[-1]

    # A sentence ending inside the cue closes it, as every subtitle writer
    # does, and so does running past the maximum display duration.
    for j in options:
        if j < widest and _ends_sentence(words[j - 1]):
            return j, j
        next_start = (words[j].start_ms if j < len(words) else end_of_audio_ms)
        if next_start - words[start].start_ms >= model.max_duration_ms:
            return j, j

    if not cps_aware:
        return widest, widest

    # Close at the widest boundary that still reads inside the ceiling, and
    # never so early that the cue is below the minimum display time.
    best: Optional[int] = None
    for j in options:
        cue = model.cue_cost(words, start, j, end_of_audio_ms,
                             lines=wrap_first_fit(words, start, j, model))
        if cue is None or cue.duration_ms < model.min_duration_ms:
            continue
        if cue.cps <= model.cps_limit:
            best = j
    # Every candidate was scored, so the rule needed the whole window.
    return (best if best is not None else widest), widest


def _greedy(words: List[Word], model: CueModel, end_of_audio_ms: int,
            cps_aware: bool) -> Segmentation:
    cues: List[Cue] = []
    horizons: List[int] = []
    start = 0
    while start < len(words):
        end, horizon = _flush_point(words, start, model, end_of_audio_ms,
                                    cps_aware)
        cues.append(_emit(words, start, end, model, end_of_audio_ms))
        horizons.append(horizon)
        start = end
    return Segmentation(cues=cues, cost=sum(c.cost for c in cues),
                        horizons=tuple(horizons))


def segment_greedy(words: List[Word], model: CueModel,
                   end_of_audio_ms: int) -> Segmentation:
    """Fill to width, flush at sentence end or maximum duration."""
    return _greedy(words, model, end_of_audio_ms, cps_aware=False)


def segment_greedy_cps(words: List[Word], model: CueModel,
                       end_of_audio_ms: int) -> Segmentation:
    """Fill to width, but stop short to respect the reading-speed ceiling."""
    return _greedy(words, model, end_of_audio_ms, cps_aware=True)
