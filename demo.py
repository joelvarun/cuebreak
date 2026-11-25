"""Segment one real stream of meeting speech three ways and print the cues.

Run:
    python demo.py
    python demo.py --stream 42 --lookahead 1500 --vtt
"""

import argparse

from cuebreak import CueModel, load_streams, segment_greedy, segment_optimal
from cuebreak.baseline import segment_greedy_cps
from cuebreak.model import break_penalty
from cuebreak.online import OnlineSegmenter
from cuebreak.vtt import to_webvtt


def show(title, cues, words, model, note=""):
    print()
    print("%s%s" % (title, note))
    print("-" * 74)
    for cue in cues:
        flag = " OVER" if cue.cps > model.cps_limit else "     "
        tail = ""
        if cue.end_index < len(words):
            if break_penalty(words[cue.end_index - 1].text,
                             words[cue.end_index].text) >= 3.0:
                tail = "   <- break strands '%s'" % words[cue.end_index - 1].text
        print("  %5.1f cps%s %5d ms" % (cue.cps, flag, cue.duration_ms))
        for line in cue.lines:
            print("    | %-42s |" % line)
        if tail:
            print("   %s" % tail)
    total = sum(c.cost for c in cues)
    print("  %d cues, total penalty %.2f" % (len(cues), total))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=int, default=None,
                        help="index into data/streams.json")
    parser.add_argument("--lookahead", type=int, default=1500,
                        help="milliseconds of audio the online segmenter buffers")
    parser.add_argument("--vtt", action="store_true",
                        help="also print the optimal segmentation as WebVTT")
    args = parser.parse_args()

    model = CueModel()
    streams = load_streams()
    if args.stream is None:
        # Default to the stream where fill-to-width and the optimum disagree
        # most, among the ones short enough to print in one screen. Pass
        # --stream to see an ordinary one instead.
        def gap(i):
            stream = streams[i]
            if not 16 <= len(stream.words) <= 30:
                return -1.0
            args_ = (stream.words, model, stream.end_of_audio_ms)
            return (segment_greedy(*args_).cost - segment_optimal(*args_).cost)
        index = max(range(len(streams)), key=gap)
    else:
        index = args.stream
    stream = streams[index]

    print("stream %d: %s, speaker %s%s"
          % (index, stream.meeting, stream.speaker,
             "" if args.stream is not None
             else "  (the stream where the rules disagree most)"))
    print("%d words, %.1f s, %.0f words per minute, %.1f characters per second"
          % (len(stream.words), stream.end_of_audio_ms / 1000.0,
             stream.words_per_minute, stream.chars_per_second))
    print()
    print("  " + stream.text)
    print()
    print("caption style: %d characters, %d lines, %.0f cps ceiling"
          % (model.max_chars, model.max_lines, model.cps_limit))

    words, end = stream.words, stream.end_of_audio_ms
    show("greedy fill to width", segment_greedy(words, model, end).cues,
         words, model)
    show("greedy fill with a reading-speed flush",
         segment_greedy_cps(words, model, end).cues, words, model)

    result = OnlineSegmenter(model, args.lookahead).run(words, end)
    delays = [max(0, d.decided_at_ms - w.start_ms)
              for d in result.decisions
              for w in words[d.cue.start_index:d.cue.end_index]]
    show("cuebreak online", result.cues, words, model,
         note=" (%d ms buffer, mean word delay %.0f ms)"
              % (args.lookahead, sum(delays) / float(len(delays))))

    best = segment_optimal(words, model, end)
    show("cuebreak offline optimum", best.cues, words, model,
         note=" (exact lower bound, needs the whole stream)")

    if args.vtt:
        print()
        print(to_webvtt(best.cues))


if __name__ == "__main__":
    main()
