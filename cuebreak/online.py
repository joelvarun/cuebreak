"""Bounded-lookahead caption segmentation.

A live captioner cannot see the end of the sentence. It has to commit to a cue
boundary while the speaker is still talking, and once a cue is on screen it
cannot be taken back. Its only freedom is delay: it may hold a cue back to
hear a few more words, at the price of captions that lag the audio.

This segmenter spends that delay well. It buffers `lookahead_ms` of audio
past the start of the cue it is building, solves the same exact shortest path
that optimal.py solves restricted to that buffer, commits only the first cue
of the window-optimal path, and slides the buffer forward. One number, the
depth of the audio buffer, is the whole configuration.

Two details make the restriction honest:

* The buffer edge is an artifact of not having heard the rest of the stream,
  so the break penalty of the cue that ends there is waived. The window must
  not be charged for a break nobody chose.
* For the same reason the buffer edge is never committed. If the whole buffer
  is window-optimal as a single cue, the buffer grows by one word and the
  window is solved again, because a cue that might still be extended cannot
  be judged.

A deeper buffer cannot appear sooner, and a buffer that spans the stream
reproduces the offline optimum exactly, so the sweep in bench.py traces the
whole quality against latency curve between the two. Every decision records
when it was taken, so the delay is measured rather than assumed.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple

from .model import Cue, CueModel, Word
from .optimal import Segmentation


class Decision(NamedTuple):
    """A committed cue plus the timing of the decision that produced it."""

    cue: Cue
    decided_at_ms: int
    #: Delay between the cue's last word being spoken and the cue appearing.
    tail_delay_ms: int
    #: Delay between the cue's first word being spoken and the cue appearing.
    cue_lag_ms: int
    buffer_boundary: int
    heard_whole_stream: bool


class OnlineResult(NamedTuple):
    cues: List[Cue]
    cost: float
    decisions: List[Decision]

    @property
    def boundaries(self) -> List[int]:
        return [c.end_index for c in self.cues]

    def delays(self) -> List[int]:
        return [d.tail_delay_ms for d in self.decisions]

    def lags(self) -> List[int]:
        return [d.cue_lag_ms for d in self.decisions]


class OnlineSegmenter:
    """Receding-horizon segmenter with a fixed word buffer."""

    def __init__(self, model: CueModel, lookahead_ms: int):
        if lookahead_ms < 0:
            raise ValueError("lookahead_ms must not be negative")
        self.model = model
        self.lookahead_ms = lookahead_ms

    def _window_path(self, words: List[Word], start: int, top: int,
                     end_of_audio_ms: int) -> Optional[int]:
        """First boundary of the cheapest segmentation of words[start:top]."""
        model = self.model
        whole_stream = top >= len(words)
        inf = float("inf")
        best: Dict[int, float] = {start: 0.0}
        first: Dict[int, int] = {}
        for k in range(start, top):
            base = best.get(k, inf)
            if base == inf:
                continue
            limit = min(model.max_end(words, k), top)
            for j in range(k + 1, limit + 1):
                cue = model.cue_cost(words, k, j, end_of_audio_ms)
                if cue is None:
                    continue
                cost = cue.cost
                if j == top and not whole_stream:
                    cost -= cue.terms["cue_break"]
                candidate = base + cost
                if candidate < best.get(j, inf):
                    best[j] = candidate
                    first[j] = j if k == start else first[k]
        return first.get(top)

    def _buffer_edge(self, words: List[Word], start: int) -> int:
        """Highest boundary inside the audio buffer for a cue starting here."""
        n = len(words)
        deadline = words[start].start_ms + self.lookahead_ms
        top = start + 1
        while top < n and words[top].start_ms <= deadline:
            top += 1
        return min(max(top, min(start + 2, n)), n)

    def _decide(self, words: List[Word], start: int,
                end_of_audio_ms: int) -> Tuple[int, int]:
        """Return (committed boundary, buffer edge used to decide)."""
        n = len(words)
        # A cue cannot hold more than this, so a cue reaching here is final
        # even though the buffer edge in general is not.
        full = min(self.model.max_end(words, start), n)
        top = self._buffer_edge(words, start)
        while True:
            boundary = self._window_path(words, start, top, end_of_audio_ms)
            if boundary is None:
                return full, top
            # A cue that stops exactly at the buffer edge might still have
            # grown, so it cannot be judged until the edge moves or the cue
            # is physically full.
            if boundary < top or top >= n or top >= full:
                return boundary, top
            top += 1

    def run(self, words: List[Word], end_of_audio_ms: int) -> OnlineResult:
        cues: List[Cue] = []
        decisions: List[Decision] = []
        n = len(words)
        start = 0
        while start < n:
            boundary, top = self._decide(words, start, end_of_audio_ms)
            cue = self.model.cue_cost(words, start, boundary, end_of_audio_ms)
            if cue is None:
                raise ValueError("no renderable cue at word %d" % start)
            decided_at = (end_of_audio_ms if top >= n
                          else words[top].start_ms)
            cues.append(cue)
            decisions.append(Decision(
                cue=cue,
                decided_at_ms=decided_at,
                tail_delay_ms=max(0, decided_at - words[boundary - 1].start_ms),
                cue_lag_ms=max(0, decided_at - words[start].start_ms),
                buffer_boundary=top,
                heard_whole_stream=top >= n))
            start = boundary
        return OnlineResult(cues=cues, cost=sum(c.cost for c in cues),
                            decisions=decisions)


def segment_online(words: List[Word], model: CueModel, end_of_audio_ms: int,
                   lookahead_ms: int) -> Segmentation:
    """Convenience wrapper returning the same shape as segment_optimal."""
    result = OnlineSegmenter(model, lookahead_ms).run(words, end_of_audio_ms)
    return Segmentation(cues=result.cues, cost=result.cost,
                        horizons=tuple(d.buffer_boundary
                                       for d in result.decisions))
