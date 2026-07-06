# model/crnn_model.py
# -------------------------------------------------
# CRNN (CNN + BiLSTM) for recognizing a flat, left-to-right sequence of math
# characters from an image, trained with CTC.
#
# Two deliberate choices versus the previous version:
#   1. The output Dense layer is LINEAR (no softmax). keras.ops.ctc_loss expects
#      raw logits; applying softmax first makes the loss wrong. Softmax/decoding
#      happens at inference time only.
#   2. Dropout is added after the dense and recurrent layers. The earlier project
#      overfit badly (in part because it had no validation signal at all); dropout
#      plus the val-monitored callbacks in the training script address that.
# -------------------------------------------------

import tensorflow as tf

from utils import IMG_HEIGHT, IMG_WIDTH, VOCAB_SIZE


def build_crnn(img_height: int = IMG_HEIGHT, img_width: int = IMG_WIDTH, vocab_size: int = VOCAB_SIZE, dropout: float = 0.3):
    inp = tf.keras.layers.Input(shape=(img_height, img_width, 1), name="image")

    # CNN feature extractor
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)          # 32 x 128

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)          # 16 x 64

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 1))(x)          # 8 x 64  (keep width/time)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)          # 8 x 64

    # Convert feature map to a width-major sequence: (time=W, H, C) -> (W, H*C)
    x = tf.keras.layers.Permute((2, 1, 3))(x)            # (64, 8, 128)
    t, h, c = x.shape[1], x.shape[2], x.shape[3]
    x = tf.keras.layers.Reshape((t, h * c))(x)           # (64, 1024)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)

    # Linear logits (blank is the last index = vocab_size). No activation.
    logits = tf.keras.layers.Dense(vocab_size + 1, name="logits")(x)

    return tf.keras.Model(inp, logits, name="crnn_ctc")
