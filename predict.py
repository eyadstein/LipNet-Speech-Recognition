import os
import sys
import difflib
import argparse

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D, LSTM, Dense, Dropout, Bidirectional,
    MaxPool3D, Activation, TimeDistributed, Flatten
)

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint.weights.h5")

# ── Vocabulary (must match training exactly) ──────────────────────
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    oov_token="",
    invert=True
)

from postprocess import correct_sentence

# ── Video loading (matches training preprocessing exactly) ────────
def load_video(path: str) -> tf.Tensor:
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video:\n{path}")

    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames could be read from video:\n{path}")

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))

    return tf.cast((frames - mean), tf.float32) / std

# ── Model architecture (must match training exactly) ───────────────
def build_model():
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'))

    return model

def predict(video_path: str) -> tuple[str, str]:
    """Returns (raw_prediction, corrected_prediction) for a given GRID-format video."""
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {CHECKPOINT_PATH}. "
            "Train the model first using LipNet.ipynb, or place a trained "
            "checkpoint.weights.h5 in the models/ folder."
        )

    model = build_model()
    model.load_weights(CHECKPOINT_PATH)

    frames = load_video(video_path)
    frames = tf.expand_dims(frames, axis=0)  # add batch dimension

    yhat = model.predict(frames, verbose=0)
    decoded = tf.keras.backend.ctc_decode(
        yhat, input_length=[75], greedy=False
    )[0][0].numpy()

    raw_pred = tf.strings.reduce_join(
        [num_to_char(word) for word in decoded[0]]
    ).numpy().decode("utf-8")

    corrected_pred = correct_sentence(raw_pred)

    return raw_pred, corrected_pred

def main():
    parser = argparse.ArgumentParser(
        description="Predict the spoken sentence from a GRID-format lipreading video."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to a GRID-format .mpg video file (75 frames, standard GRID framing)."
    )
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: video file not found: {args.video}")
        sys.exit(1)

    print(f"Loading and processing: {args.video}")
    raw_pred, corrected_pred = predict(args.video)

    print("=" * 60)
    print(f"Raw prediction:       {raw_pred}")
    print(f"Corrected prediction: {corrected_pred}")
    print("=" * 60)

if __name__ == "__main__":
    main()
