# inference/decode.py
# -------------------------------------------------
# Shared CTC greedy decoder used by BOTH training-time evaluation and inference,
# so "how the model is scored" and "what the app shows" can never diverge.
#
# Semantics verified empirically against this exact Keras version:
#   - the model outputs raw logits with the CTC blank as the LAST index
#     (index == VOCAB_SIZE), so mask_index=VOCAB_SIZE.
#   - keras.ops.ctc_decode collapses repeated symbols and returns -1 padding.
# -------------------------------------------------

import keras
import numpy as np

from utils import VOCAB_SIZE, idx_to_char


def greedy_decode(logits: np.ndarray) -> list[str]:
    """logits: (batch, time, VOCAB_SIZE + 1). Returns a list of decoded strings."""
    batch, time = logits.shape[0], logits.shape[1]
    seq_lengths = np.full((batch,), time, dtype="int32")
    decoded, _ = keras.ops.ctc_decode(
        logits, sequence_lengths=seq_lengths, strategy="greedy", mask_index=VOCAB_SIZE
    )
    decoded = np.array(decoded[0])  # (batch, max_decoded_len), -1 padded

    results = []
    for row in decoded:
        chars = [idx_to_char[int(i)] for i in row if 0 <= int(i) < VOCAB_SIZE]
        results.append("".join(chars))
    return results
