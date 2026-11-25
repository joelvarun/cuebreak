# Corpus attribution

`streams.json` holds word timings derived from the **AMI Meeting Corpus**
manual annotations (`ami_public_manual_1.6.2`), meetings EN2001a, ES2002a,
IS1000a and TS3003a. Only the word strings and their millisecond offsets are
kept; no audio and none of the other annotation layers are redistributed here.

- Corpus: https://groups.inf.ed.ac.uk/ami/corpus/
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/
- Citation: J. Carletta, S. Ashby, S. Bourban, M. Flynn, M. Guillemot,
  T. Hain, J. Kadlec, V. Karaiskos, W. Kraaij, M. Kronenthal, G. Lathoud,
  M. Lincoln, A. Lisowska, I. McCowan, W. Post, D. Reidsma and P. Wellner.
  "The AMI Meeting Corpus: A Pre-Announcement."
  Machine Learning for Multimodal Interaction (MLMI), 2005, pages 28 to 39.

Regenerate from the upstream archive with:

    python tools/build_streams.py
