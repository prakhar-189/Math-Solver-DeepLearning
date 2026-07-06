# training/training_crnn.py
# -------------------------------------------------
# Trains the CRNN+CTC recognizer on the synthetic flat-arithmetic dataset and
# evaluates it on BOTH a synthetic validation split and the real CROHME
# flat-arithmetic test set.
#
# Key fixes over the previous version of this project:
#   - There is a real validation split, and EarlyStopping / ReduceLROnPlateau /
#     ModelCheckpoint all monitor VALIDATION loss. The old script monitored
#     training loss only, so overfitting was invisible and unpreventable.
#   - Images are fed through the SAME utils.preprocess_image used at inference.
#   - Uses keras.ops.ctc_loss with the blank as the last index (verified).
# -------------------------------------------------

import argparse
import json
import os

import cv2
import keras
import numpy as np
import tensorflow as tf

from inference.decode import greedy_decode
from model.crnn_model import build_crnn
from utils import VOCAB_SIZE, encode_label, preprocess_image

MAX_LABEL_LEN = 24
PAD = -1

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_split(labels_path: str, img_dir: str, max_items: int | None = None):
    """Load a split into (X, y_padded, raw_labels). Images preprocessed with the
    canonical pipeline; labels encoded to indices and right-padded with PAD."""
    X, y, raw = [], [], []
    with open(labels_path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if max_items:
        lines = lines[:max_items]
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) < 2:
            continue
        name, label = parts[0], parts[1]
        if len(label) > MAX_LABEL_LEN:
            continue  # skip labels longer than the training budget
        img = cv2.imread(os.path.join(img_dir, name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        X.append(preprocess_image(img)[:, :, 0])
        enc = encode_label(label)
        enc = enc + [PAD] * (MAX_LABEL_LEN - len(enc))
        y.append(enc)
        raw.append(label)
    X = np.expand_dims(np.array(X, dtype="float32"), -1)
    y = np.array(y, dtype="int32")
    return X, y, raw


def ctc_loss_fn(y_true, y_pred):
    """CTC loss. y_true: (batch, MAX_LABEL_LEN) padded with PAD(-1). y_pred: logits."""
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.not_equal(y_true, PAD)
    target_length = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
    target = tf.where(mask, y_true, tf.zeros_like(y_true))  # pad -> 0 (ignored past length)

    batch = tf.shape(y_pred)[0]
    time = tf.shape(y_pred)[1]
    output_length = tf.fill((batch,), time)

    return keras.ops.ctc_loss(target, y_pred, target_length, output_length, mask_index=VOCAB_SIZE)


def _levenshtein(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[-1]


def evaluate(model, X, raw_labels, name: str, batch_size: int = 64):
    preds = []
    for i in range(0, len(X), batch_size):
        logits = model.predict(X[i:i + batch_size], verbose=0)
        preds.extend(greedy_decode(logits))
    exact = sum(p == t for p, t in zip(preds, raw_labels))
    total_chars = sum(len(t) for t in raw_labels)
    total_edits = sum(_levenshtein(p, t) for p, t in zip(preds, raw_labels))
    result = {
        "set": name,
        "n": len(raw_labels),
        "exact_match_accuracy": exact / len(raw_labels) if raw_labels else 0.0,
        "char_error_rate": total_edits / total_chars if total_chars else 0.0,
    }
    print(f"  [{name}] n={result['n']}  exact-match={result['exact_match_accuracy']:.4f}  CER={result['char_error_rate']:.4f}")
    return result, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train", type=int, default=None, help="cap training images (for smoke tests)")
    parser.add_argument("--model-out", default=os.path.join(ROOT, "math_solver_crnn.keras"))
    parser.add_argument("--metrics-out", default=os.path.join(ROOT, "eval_metrics.json"))
    args = parser.parse_args()

    print("Loading data...")
    Xtr, ytr, _ = load_split(
        os.path.join(ROOT, "data/synthetic/train/labels.txt"),
        os.path.join(ROOT, "data/synthetic/train/img"),
        max_items=args.max_train,
    )
    Xva, yva, raw_va = load_split(
        os.path.join(ROOT, "data/synthetic/val/labels.txt"),
        os.path.join(ROOT, "data/synthetic/val/img"),
    )
    print(f"  train={len(Xtr)}  val={len(Xva)}")

    model = build_crnn()
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr, clipnorm=1.0), loss=ctc_loss_fn)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(args.model_out, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    print("Training...")
    model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    print("\nEvaluating...")
    metrics = {}
    res_val, _ = evaluate(model, Xva, raw_va, "synthetic_val")
    metrics["synthetic_val"] = res_val

    real_labels_path = os.path.join(ROOT, "data/real_test/labels.txt")
    if os.path.exists(real_labels_path):
        Xte, _, raw_te = load_split(real_labels_path, os.path.join(ROOT, "data/real_test/img"))
        res_real, _ = evaluate(model, Xte, raw_te, "real_crohme_test")
        metrics["real_crohme_test"] = res_real

    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model -> {args.model_out}")
    print(f"Saved metrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()
