"""Cue cost model for timed captions.

A cue is a contiguous run of spoken words shown on screen as one or two lines.
The quality constraints below follow published subtitling guidance (EBU-TT-D
and the BBC and Netflix timed-text style guides): a reading-speed ceiling in
characters per second, a minimum and maximum on-screen duration, a line length
ceiling, at most two lines, balanced line lengths, and line breaks placed at
syntactic boundaries rather than inside a phrase.

Every term is a function of the cue alone, which is what makes exact
segmentation possible: see optimal.py.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

#: Function words that must not end a line, because the reader is left holding
#: a determiner or preposition with nothing to attach it to.
NO_BREAK_AFTER = frozenset("""
a an the this that these those my your his her its our their
of to in on at by for with from into onto upon over under
and or nor but so yet as if than because while whereas
is are was were be been being am do does did has have had
no not very more most such each every both either neither
""".split())

#: Words that prefer to start a line, so breaking just before them is cheap.
BREAK_BEFORE_BONUS = frozenset("""
and or but so because while although though if when whereas
that which who whom whose where after before since unless
""".split())

STRONG_PUNCT = (".", "!", "?")
WEAK_PUNCT = (",", ";", ":")
CLOSERS = "\"')]"


@dataclass(frozen=True)
class Word:
    """One spoken word with the audio time at which it starts."""

    text: str
    start_ms: int

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass
class Cue:
    """A rendered cue: words [start, end) laid out over one or two lines."""

    start_index: int
    end_index: int
    lines: List[str]
    start_ms: int
    end_ms: int
    cost: float = 0.0
    terms: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def chars(self) -> int:
        return sum(len(ln) for ln in self.lines)

    @property
    def cps(self) -> float:
        """Reading speed in characters per second."""
        if self.duration_ms <= 0:
            return float("inf")
        return self.chars * 1000.0 / self.duration_ms


def _bare(token: str) -> str:
    return token.strip(CLOSERS).lower()


def break_penalty(left: str, right: Optional[str]) -> float:
    """Cost of placing a break between word `left` and word `right`.

    Zero after sentence-final punctuation, small after a comma, large after a
    determiner or preposition. `right` is None at the end of the transcript.
    """
    stripped = left.rstrip(CLOSERS)
    if stripped.endswith(STRONG_PUNCT):
        return 0.0
    if stripped.endswith(WEAK_PUNCT):
        return 0.4
    if right is None:
        return 0.0
    if _bare(left).rstrip(".,;:") in NO_BREAK_AFTER:
        return 3.0
    if _bare(right) in BREAK_BEFORE_BONUS:
        return 0.6
    if left.endswith("-") or right.startswith("-"):
        return 3.0
    return 1.4


@dataclass
class CueModel:
    """Constraints and penalty weights for one caption style.

    The limits follow the Netflix English timed-text style guide: 42
    characters per line, at most two lines, a 17 character-per-second reading
    speed ceiling, and a minimum display time of five sixths of a second.

    Every penalty is linear in its own natural unit, so no single term can
    swamp the others by being squared. The weights encode the priority order
    the style guides state: keeping a phrase intact comes first, then reading
    speed, then display time, then line balance. One dangling break therefore
    costs 6.0, which is more than a 10 character-per-second overrun (5.0) and
    far more than the worst possible imbalance (2.0).
    """

    max_chars: int = 42
    max_lines: int = 2
    cps_limit: float = 17.0
    min_duration_ms: int = 833
    max_duration_ms: int = 7000

    #: Per character-per-second above the ceiling.
    w_cps: float = 0.5
    #: Per second below the minimum display time.
    w_duration: float = 4.0
    #: Per unit of line-length difference, as a fraction of max_chars.
    w_balance: float = 2.0
    #: Multiplies break_penalty, which runs from 0.0 to 3.0.
    w_cue_break: float = 2.0
    w_line_break: float = 1.0
    #: Charged once per cue, so the segmenter does not flicker for free.
    w_cue: float = 0.5
    #: Charged per sentence end buried inside a cue.
    w_sentence_mix: float = 0.8

    @property
    def max_cue_chars(self) -> int:
        return self.max_lines * self.max_chars + (self.max_lines - 1)

    def display_end_ms(self, start_ms: int, next_start_ms: int) -> int:
        """When the cue comes off screen: replaced, or timed out."""
        return min(next_start_ms, start_ms + self.max_duration_ms)

    def layout_cost(self, words: List[Word], i: int,
                    lines: List[str]) -> float:
        """Penalty for one specific rendering of a cue starting at word i.

        Combines line-length imbalance with the quality of the break between
        the two lines. A one line cue has no layout penalty.
        """
        if len(lines) < 2:
            return 0.0
        if len(lines) > self.max_lines:
            raise ValueError("cue needs %d lines, the caption area holds %d"
                             % (len(lines), self.max_lines))
        imbalance = abs(len(lines[0]) - len(lines[1])) / float(self.max_chars)
        cut = i + len(lines[0].split())
        return (self.w_balance * imbalance
                + self.w_line_break * break_penalty(words[cut - 1].text,
                                                    words[cut].text))

    def layout(self, words: List[Word], i: int, j: int
               ) -> Optional[Tuple[List[str], float]]:
        """Cheapest one or two line layout of words[i:j], or None if it cannot fit."""
        texts = [w.text for w in words[i:j]]
        whole = " ".join(texts)
        best: Optional[Tuple[List[str], float]] = None
        if len(whole) <= self.max_chars:
            best = ([whole], 0.0)
        if self.max_lines < 2:
            return best
        for cut in range(1, len(texts)):
            top = " ".join(texts[:cut])
            bottom = " ".join(texts[cut:])
            if len(top) > self.max_chars or len(bottom) > self.max_chars:
                continue
            cost = self.layout_cost(words, i, [top, bottom])
            if best is None or cost < best[1]:
                best = ([top, bottom], cost)
        return best

    def cue_cost(self, words: List[Word], i: int, j: int, end_of_audio_ms: int,
                 lines: Optional[List[str]] = None) -> Optional[Cue]:
        """Cost of showing words[i:j] as a single cue, or None if infeasible.

        The cue stays on screen until the next cue starts, which is the start
        time of word j, so the cost depends only on i and j. Pass `lines` to
        score a rendering someone else chose, instead of the cheapest one.
        """
        if lines is None:
            laid = self.layout(words, i, j)
            if laid is None:
                return None
            lines, layout_cost = laid
        else:
            layout_cost = self.layout_cost(words, i, lines)

        start_ms = words[i].start_ms
        next_start = words[j].start_ms if j < len(words) else end_of_audio_ms
        end_ms = self.display_end_ms(start_ms, next_start)
        duration_ms = max(1, end_ms - start_ms)

        chars = sum(len(ln) for ln in lines)
        cps = chars * 1000.0 / duration_ms
        cps_over = max(0.0, cps - self.cps_limit)
        short_by_s = max(0.0, self.min_duration_ms - duration_ms) / 1000.0

        right = words[j].text if j < len(words) else None
        cue_break = break_penalty(words[j - 1].text, right)

        internal = sum(1 for k in range(i, j - 1)
                       if words[k].text.rstrip(CLOSERS).endswith(STRONG_PUNCT))

        terms = {
            "cps": self.w_cps * cps_over,
            "duration": self.w_duration * short_by_s,
            "layout": layout_cost,
            "cue_break": self.w_cue_break * cue_break,
            "sentence_mix": self.w_sentence_mix * internal,
            "cue": self.w_cue,
        }
        return Cue(start_index=i, end_index=j, lines=lines, start_ms=start_ms,
                   end_ms=end_ms, cost=sum(terms.values()), terms=terms)

    def max_end(self, words: List[Word], i: int) -> int:
        """Largest j such that words[i:j] can still fit in one cue.

        Bounds the arc set of the segmentation graph, and bounds how far ahead
        an online segmenter has to look before it can decide anything.
        """
        total, j = 0, i
        while j < len(words):
            total += words[j].length + (1 if j > i else 0)
            if total > self.max_cue_chars:
                break
            j += 1
        return max(i + 1, j)
