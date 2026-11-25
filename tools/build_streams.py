"""Extract word-aligned caption streams from the AMI Meeting Corpus.

The AMI manual annotations carry a start and end time for every word of real
meeting speech, which is exactly what a live captioner sees arriving from a
streaming recogniser. Meeting captioning is also the setting where caption
delay actually matters, so the corpus matches the problem.

Source:   https://groups.inf.ed.ac.uk/ami/corpus/
Licence:  Creative Commons Attribution 4.0 International (CC BY 4.0)
Citation: J. Carletta et al., "The AMI Meeting Corpus: A pre-announcement",
          MLMI 2005.

Run:
    python tools/build_streams.py
Downloads ami_public_manual_1.6.2.zip once into data/, then writes
data/streams.json: the derived word timings only, not the annotations.
"""

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_PATH = os.path.join(HERE, "data", "ami_public_manual_1.6.2.zip")
OUT = os.path.join(HERE, "data", "streams.json")
ZIP_URL = ("https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/"
           "ami_public_manual_1.6.2.zip")

#: One meeting from each of the four AMI recording series, so the corpus
#: spans different rooms, speakers and first languages.
MEETINGS = ["ES2002a", "IS1000a", "TS3003a", "EN2001a"]

#: A silence longer than this ends the caption stream: the speaker stopped,
#: so there is nothing to hold a cue open for.
STREAM_GAP_MS = 700
MAX_STREAM_WORDS = 60
MIN_STREAM_WORDS = 5

NITE = "{http://nite.sourceforge.net/}"


def ensure_zip():
    if os.path.exists(ZIP_PATH):
        return
    print("downloading AMI annotations (23 MB) ...")
    code = subprocess.call(["curl", "-sL", "--max-time", "600",
                            "-o", ZIP_PATH, ZIP_URL])
    if code != 0 or not os.path.exists(ZIP_PATH):
        sys.exit("download failed; fetch %s into data/ by hand" % ZIP_URL)


def read_words(archive, entry):
    """Yield (text, start_ms, end_ms, is_punctuation) in transcript order."""
    root = ET.fromstring(archive.read(entry))
    for node in root:
        tag = node.tag.split("}")[-1]
        if tag != "w":
            continue
        text = (node.text or "").strip()
        start, end = node.get("starttime"), node.get("endtime")
        if not text or start is None or end is None:
            continue
        yield (text, int(round(float(start) * 1000)),
               int(round(float(end) * 1000)), node.get("punc") == "true")


def to_tokens(raw):
    """Attach punctuation to the word before it, as a caption would render it."""
    tokens = []
    for text, start_ms, end_ms, is_punc in raw:
        if is_punc and tokens:
            tokens[-1]["text"] += text
            continue
        if is_punc:
            continue
        tokens.append({"text": text, "start_ms": start_ms, "end_ms": end_ms})
    return tokens


def split_streams(tokens):
    """Cut the token sequence at long silences and at a length ceiling."""
    streams, current = [], []
    for token in tokens:
        if current:
            silence = token["start_ms"] - current[-1]["end_ms"]
            if silence > STREAM_GAP_MS or len(current) >= MAX_STREAM_WORDS:
                streams.append(current)
                current = []
        current.append(token)
    if current:
        streams.append(current)
    return [s for s in streams if len(s) >= MIN_STREAM_WORDS]


def main():
    ensure_zip()
    archive = zipfile.ZipFile(ZIP_PATH)
    entries = [n for n in archive.namelist()
               if n.startswith("words/") and n.endswith(".words.xml")
               and n.split("/")[1].split(".")[0] in MEETINGS]
    out = []
    for entry in sorted(entries):
        meeting, speaker = entry.split("/")[1].split(".")[:2]
        tokens = to_tokens(read_words(archive, entry))
        streams = split_streams(tokens)
        for stream in streams:
            start = stream[0]["start_ms"]
            out.append({
                "meeting": meeting,
                "speaker": speaker,
                "words": [[t["text"], t["start_ms"] - start] for t in stream],
                "end_ms": stream[-1]["end_ms"] - start,
            })
        print("%-14s speaker %s  %5d words -> %3d streams"
              % (meeting, speaker, len(tokens), len(streams)))

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    words = sum(len(s["words"]) for s in out)
    seconds = sum(s["end_ms"] for s in out) / 1000.0
    print("%d streams, %d words, %.1f s of speech, %.0f words per minute"
          % (len(out), words, seconds, words / (seconds / 60.0)))
    print("wrote", OUT, "(%.1f KB)" % (os.path.getsize(OUT) / 1024.0))


if __name__ == "__main__":
    main()
