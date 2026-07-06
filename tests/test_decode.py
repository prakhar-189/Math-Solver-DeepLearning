import numpy as np

from inference.decode import greedy_decode
from utils import VOCAB_SIZE, char_to_idx


def _peaked_logits(seq_indices, time, num_classes):
    """Build a (1, time, num_classes) logit tensor strongly peaked at the given
    per-timestep class indices."""
    logits = np.full((1, time, num_classes), -10.0, dtype="float32")
    for t, c in enumerate(seq_indices):
        logits[0, t, c] = 10.0
    return logits


def test_greedy_decode_collapses_repeats_and_strips_blank():
    num_classes = VOCAB_SIZE + 1
    blank = VOCAB_SIZE
    # "1", "1", "+", blank, "2" -> collapses repeated 1s, drops blank -> "11+2"? No:
    # repeated identical adjacent labels collapse to ONE, so [1,1] -> single "1".
    seq = [char_to_idx["1"], char_to_idx["1"], char_to_idx["+"], blank, char_to_idx["2"]]
    logits = _peaked_logits(seq, time=len(seq), num_classes=num_classes)
    out = greedy_decode(logits)[0]
    assert out == "1+2"


def test_greedy_decode_double_char_via_blank_separation():
    num_classes = VOCAB_SIZE + 1
    blank = VOCAB_SIZE
    # To emit "11", a blank must separate the two 1s: [1, blank, 1]
    seq = [char_to_idx["1"], blank, char_to_idx["1"]]
    logits = _peaked_logits(seq, time=len(seq), num_classes=num_classes)
    assert greedy_decode(logits)[0] == "11"


def test_greedy_decode_batch():
    num_classes = VOCAB_SIZE + 1
    blank = VOCAB_SIZE
    seqs = [
        [char_to_idx["3"], blank, char_to_idx["+"], blank, char_to_idx["4"]],
        [char_to_idx["9"], blank, char_to_idx["-"], blank, char_to_idx["0"]],
    ]
    t = 5
    logits = np.full((2, t, num_classes), -10.0, dtype="float32")
    for b, seq in enumerate(seqs):
        for ti, c in enumerate(seq):
            logits[b, ti, c] = 10.0
    out = greedy_decode(logits)
    assert out == ["3+4", "9-0"]
