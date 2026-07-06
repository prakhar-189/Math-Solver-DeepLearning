# inference/predict.py
# -------------------------------------------------
# Inference pipeline: image -> recognized expression -> (optional) solved result.
# Uses the SAME utils.preprocess_image and inference.decode.greedy_decode that
# training uses, so what the app sees is exactly what the model was scored on.
# -------------------------------------------------

import os

import numpy as np
import sympy as sp
import tensorflow as tf

from inference.decode import greedy_decode
from utils import normalize_equation, preprocess_image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_ROOT, "math_solver_crnn.keras")

_model = None


def get_model():
    """Lazily load the trained model (so importing this module is cheap)."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train it first: python -m training.training_crnn"
            )
        # compile=False: we only need forward passes; the custom CTC loss isn't required for inference.
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model


def predict_equation(image: np.ndarray) -> str:
    """Recognize the handwritten expression in a grayscale/BGR image."""
    x = preprocess_image(image, add_batch_dim=True)
    logits = get_model().predict(x, verbose=0)
    equation = greedy_decode(logits)[0]
    return normalize_equation(equation)


def solve_equation(equation: str):
    """Evaluate a recognized expression. Returns (result_str, error_str).

    Handles both a bare expression ("(4+3)*8" -> 56) and an equality
    ("2+2=4" -> reports whether the two sides match)."""
    if not equation:
        return None, "No expression recognized."
    try:
        if "=" in equation:
            lhs, rhs = equation.split("=", 1)
            lhs_v = sp.sympify(lhs)
            rhs_v = sp.sympify(rhs) if rhs.strip() else None
            if rhs_v is None:
                return str(lhs_v), None
            ok = sp.simplify(lhs_v - rhs_v) == 0
            return f"{lhs_v} {'=' if ok else '!='} {rhs_v}  ({'correct' if ok else 'incorrect'})", None
        value = sp.sympify(equation)
        return str(value), None
    except Exception as e:  # noqa: BLE001 - surface any parse/eval error to the UI
        return None, f"Could not evaluate '{equation}': {e}"
