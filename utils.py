# utils.py
# -------------------------------------------------
# Vocabulary + a SINGLE canonical image-preprocessing function shared by
# training, evaluation, and inference.
#
# The previous version of this project had two different preprocessing paths
# gated on an `is_training` flag: training saw raw resized images, while
# inference ran an entirely separate crop/threshold/erode pipeline. The model
# was therefore evaluated on a distribution it had never trained on, which is
# the core reason the Streamlit app produced garbage. There is now ONE path,
# used everywhere, so train-time and inference-time inputs are identical.
# -------------------------------------------------

import cv2
import numpy as np

# Canonical, fixed vocabulary for flat arithmetic. Defined in code (not read
# from a vocab.txt that could drift out of sync with a saved model). "*" and "/"
# are the multiply/divide CLASSES; they are rendered/handwritten as "×"/"÷" but
# always labeled with these ASCII forms so labels stay sympy-evaluable.
VOCAB = "0123456789+-*/().="
char_to_idx = {c: i for i, c in enumerate(VOCAB)}
idx_to_char = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
BLANK_TOKEN = VOCAB_SIZE  # CTC blank is the last index

IMG_HEIGHT = 64
IMG_WIDTH = 256


def _normalize_polarity(gray: np.ndarray) -> np.ndarray:
    """Return a black-strokes-on-white-background image regardless of the input
    polarity. Synthetic renders are black-on-white; real CROHME images are
    white-on-black. Standardizing here means both look the same to the model."""
    if gray.mean() < 127:  # mostly dark => strokes are light on a dark background
        gray = 255 - gray
    return gray


def _crop_to_ink(gray: np.ndarray) -> np.ndarray:
    """Crop to the bounding box of the ink (dark pixels) with a small margin.
    Makes the model robust to arbitrary margins in uploaded photos, and matches
    the tight crops of the real CROHME images."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return gray
    x, y, w, h = cv2.boundingRect(coords)
    m = max(2, int(0.08 * h))
    x0 = max(0, x - m)
    y0 = max(0, y - m)
    x1 = min(gray.shape[1], x + w + m)
    y1 = min(gray.shape[0], y + h + m)
    return gray[y0:y1, x0:x1]


def preprocess_image(image: np.ndarray, add_batch_dim: bool = False) -> np.ndarray:
    """Canonical preprocessing used by training, evaluation, and inference.

    image: grayscale or BGR uint8 array.
    Returns float32 array in [0, 1] of shape (IMG_HEIGHT, IMG_WIDTH, 1),
    or (1, IMG_HEIGHT, IMG_WIDTH, 1) if add_batch_dim=True (for single-image
    inference).
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = _normalize_polarity(image)
    image = _crop_to_ink(image)

    h, w = image.shape
    new_w = int(IMG_HEIGHT * w / h) if h > 0 else IMG_WIDTH
    new_w = max(1, min(new_w, IMG_WIDTH))
    image = cv2.resize(image, (new_w, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

    canvas = np.full((IMG_HEIGHT, IMG_WIDTH), 255, dtype=np.uint8)  # white padding
    canvas[:, :new_w] = image

    arr = canvas.astype("float32") / 255.0
    arr = np.expand_dims(arr, -1)
    if add_batch_dim:
        arr = np.expand_dims(arr, 0)
    return arr


def encode_label(label: str) -> list[int]:
    """Map a label string to a list of vocab indices (unknown chars dropped)."""
    return [char_to_idx[c] for c in label if c in char_to_idx]


def normalize_equation(eq: str) -> str:
    """Map any stray unicode operator glyphs to ASCII for sympy evaluation."""
    return (
        eq.replace("×", "*").replace("·", "*").replace("x", "*")
        .replace("÷", "/")
        .replace("−", "-")
    )
