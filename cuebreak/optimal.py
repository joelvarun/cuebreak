"""Exact caption segmentation by shortest path over cue boundaries.

Because a cue stays on screen until the next cue starts, the cost of a cue
covering words[i:j] depends only on i and j (see model.CueModel.cue_cost).
The segmentation problem is therefore a shortest path in a directed acyclic
graph whose nodes are word boundaries 0..n and whose arcs are feasible cues.
Dijkstra is unnecessary: the graph is topologically ordered by index, so a
single forward pass gives the exact minimum total penalty.

The arc set is bounded: a cue holds at most max_lines * max_chars characters,
so from boundary i only boundaries up to model.max_end(words, i) are
reachable. The pass is therefore O(n * w) in the number of words, with w the
largest number of words that fit in one cue.
"""

from typing import List, NamedTuple, Optional, Tuple

from .model import Cue, CueModel, Word


class Segmentation(NamedTuple):
    cues: List[Cue]
    cost: float
    #: For each cue, the index of the last word the method had to observe
    #: before it could commit. Every method needs the word after the closing
    #: boundary, because that is when the cue comes off screen, so this is
    #: never below the boundary itself.
    horizons: Tuple[int, ...] = ()

    @property
    def boundaries(self) -> List[int]:
        return [c.end_index for c in self.cues]


def _shortest_path(words: List[Word], model: CueModel, end_of_audio_ms: int):
    """Return (best_cost, predecessor, arc) arrays over boundaries 0..n."""
    n = len(words)
    inf = float("inf")
    best = [inf] * (n + 1)
    prev: List[int] = [-1] * (n + 1)
    arc: List[Optional[Cue]] = [None] * (n + 1)
    best[0] = 0.0

    for i in range(n):
        if best[i] == inf:
            continue
        limit = model.max_end(words, i)
        for j in range(i + 1, min(limit, n) + 1):
            cue = model.cue_cost(words, i, j, end_of_audio_ms)
            if cue is None:
                continue
            candidate = best[i] + cue.cost
            if candidate < best[j]:
                best[j] = candidate
                prev[j] = i
                arc[j] = cue
    return best, prev, arc


def _trace(prev, arc, end: int) -> List[Cue]:
    cues: List[Cue] = []
    node = end
    while node > 0:
        cue = arc[node]
        if cue is None:
            raise ValueError("no feasible segmentation reaches boundary %d" % node)
        cues.append(cue)
        node = prev[node]
    cues.reverse()
    return cues


def segment_optimal(words: List[Word], model: CueModel,
                    end_of_audio_ms: int) -> Segmentation:
    """Minimum total penalty segmentation of a whole word stream.

    This is the certificate the online segmenter is measured against: no
    causal or non-causal policy using this cost model can score lower.
    """
    if not words:
        return Segmentation(cues=[], cost=0.0, horizons=())
    best, prev, arc = _shortest_path(words, model, end_of_audio_ms)
    n = len(words)
    if best[n] == float("inf"):
        raise ValueError("stream has no feasible segmentation under this model")
    cues = _trace(prev, arc, n)
    return Segmentation(cues=cues, cost=best[n], horizons=(n,) * len(cues))


def suffix_lower_bounds(words: List[Word], model: CueModel,
                        end_of_audio_ms: int) -> List[float]:
    """Exact cost to finish the stream from each boundary.

    Used only by the analysis tools: it needs the whole stream, so an online
    segmenter cannot call it.
    """
    n = len(words)
    inf = float("inf")
    tail = [inf] * (n + 1)
    tail[n] = 0.0
    for i in range(n - 1, -1, -1):
        limit = min(model.max_end(words, i), n)
        for j in range(i + 1, limit + 1):
            if tail[j] == inf:
                continue
            cue = model.cue_cost(words, i, j, end_of_audio_ms)
            if cue is not None:
                tail[i] = min(tail[i], cue.cost + tail[j])
    return tail
