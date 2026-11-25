"""Render a segmentation as WebVTT."""

from typing import Iterable, List

from .model import Cue


def _stamp(ms: int) -> str:
    ms = max(0, int(ms))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d.%03d" % (hours, minutes, seconds, ms)


def to_webvtt(cues: Iterable[Cue], offset_ms: int = 0) -> str:
    """Serialise cues as a WebVTT document."""
    out: List[str] = ["WEBVTT", ""]
    for index, cue in enumerate(cues, start=1):
        out.append(str(index))
        out.append("%s --> %s" % (_stamp(cue.start_ms + offset_ms),
                                  _stamp(cue.end_ms + offset_ms)))
        out.extend(cue.lines)
        out.append("")
    return "\n".join(out)
