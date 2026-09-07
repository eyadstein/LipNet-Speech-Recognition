"""Post-processing utilities for LipNet predictions.

Since the GRID corpus uses a small, fixed vocabulary, raw CTC predictions
can be corrected against that known word list to fix decoding artifacts
(e.g. double-letter collapse in words like "green" -> "gren").
"""

import difflib

GRID_VOCAB = [
    "bin", "lay", "place", "set",
    "blue", "green", "red", "white",
    "at", "by", "in", "with",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
    "u", "v", "x", "y", "z",
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
    "again", "now", "please", "soon",
]


def correct_sentence(sentence: str, vocab: list = GRID_VOCAB, cutoff: float = 0.5) -> str:
    """Correct a predicted sentence by snapping each word to its nearest match
    in a known fixed vocabulary."""
    words = sentence.strip().split()
    corrected = []
    for word in words:
        if word in vocab:
            corrected.append(word)
        else:
            match = difflib.get_close_matches(word, vocab, n=1, cutoff=cutoff)
            corrected.append(match[0] if match else word)
    return " ".join(corrected)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) between a reference and hypothesis sentence."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return d[len(ref_words)][len(hyp_words)] / max(len(ref_words), 1)
