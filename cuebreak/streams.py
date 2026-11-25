"""Load word-aligned caption streams produced by tools/build_streams.py.

Each stream is one uninterrupted run of speech from one speaker: the words
arrive in order, with the millisecond offset at which each was spoken, and the
stream ends when the speaker pauses. That is what a live captioner has to work
with.
"""

import json
import os
from typing import List, NamedTuple, Sequence, Tuple

from .model import Word

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(HERE, "data", "streams.json")


class Stream(NamedTuple):
    meeting: str
    speaker: str
    words: List[Word]
    end_of_audio_ms: int

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def words_per_minute(self) -> float:
        if self.end_of_audio_ms <= 0:
            return 0.0
        return len(self.words) / (self.end_of_audio_ms / 60000.0)

    @property
    def chars_per_second(self) -> float:
        if self.end_of_audio_ms <= 0:
            return 0.0
        return len(self.text) * 1000.0 / self.end_of_audio_ms


def load(path: str = DEFAULT_PATH) -> List[Stream]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    out = []
    for item in raw:
        words = [Word(text=text, start_ms=int(start))
                 for text, start in item["words"]]
        end = max(int(item["end_ms"]), words[-1].start_ms + 1)
        out.append(Stream(meeting=item["meeting"], speaker=item["speaker"],
                          words=words, end_of_audio_ms=end))
    return out


def rate_buckets(streams: Sequence[Stream], labels=("slow", "medium", "fast")
                 ) -> List[Tuple[str, List[Stream]]]:
    """Split streams into equal-size groups by their own speaking rate.

    The groups are real speakers talking at their natural pace, so the
    reading-speed pressure on the captioner is not a parameter anyone chose.
    """
    ordered = sorted(streams, key=lambda s: s.words_per_minute)
    size = len(ordered) // len(labels)
    out = []
    for index, label in enumerate(labels):
        lo = index * size
        hi = len(ordered) if index == len(labels) - 1 else lo + size
        out.append((label, ordered[lo:hi]))
    return out
