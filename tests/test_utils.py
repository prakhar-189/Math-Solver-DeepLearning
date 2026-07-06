import numpy as np

from utils import (
    IMG_HEIGHT,
    IMG_WIDTH,
    VOCAB,
    VOCAB_SIZE,
    encode_label,
    normalize_equation,
    preprocess_image,
)


def test_vocab_indices_are_contiguous():
    assert VOCAB_SIZE == len(VOCAB)
    assert encode_label("12+3") == [VOCAB.index(c) for c in "12+3"]


def test_encode_label_drops_unknown_chars():
    # 'a' is not in the arithmetic vocab and should be dropped
    assert encode_label("1a2") == [VOCAB.index("1"), VOCAB.index("2")]


def test_preprocess_output_shape_and_range():
    img = np.full((40, 90), 255, np.uint8)
    img[10:30, 10:80] = 0  # some black ink
    out = preprocess_image(img)
    assert out.shape == (IMG_HEIGHT, IMG_WIDTH, 1)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_preprocess_add_batch_dim():
    img = np.full((40, 90), 255, np.uint8)
    img[10:30, 10:80] = 0
    out = preprocess_image(img, add_batch_dim=True)
    assert out.shape == (1, IMG_HEIGHT, IMG_WIDTH, 1)


def test_preprocess_polarity_normalization():
    """A black-on-white image and its white-on-black inverse should preprocess to
    the same canonical representation (the fix for train/inference polarity mismatch)."""
    black_on_white = np.full((40, 90), 255, np.uint8)
    black_on_white[15:25, 20:70] = 0
    white_on_black = 255 - black_on_white

    a = preprocess_image(black_on_white)
    b = preprocess_image(white_on_black)
    # Near-identical after canonical normalization (allow tiny res/threshold diffs).
    assert np.mean(np.abs(a - b)) < 0.02


def test_normalize_equation_maps_unicode_operators():
    assert normalize_equation("4×3") == "4*3"
    assert normalize_equation("8÷2") == "8/2"
    assert normalize_equation("5−1") == "5-1"
