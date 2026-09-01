# LipNet — Sentence-Level Lipreading from Silent Video

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)

## Overview

LipNet is a deep learning system that performs **sentence-level lipreading**: given a silent video of a person speaking, it predicts the full sentence being spoken, without any audio input. This repository implements a Conv3D + Bidirectional LSTM architecture trained end-to-end with Connectionist Temporal Classification (CTC) loss to map variable-length video sequences directly to sentences.

Rather than classifying isolated words or relying on hand-crafted visual features and separate language models, the network learns spatiotemporal visual features and character-level sequence alignment jointly, in a single trainable pipeline.

## Why this problem is hard

- **No audio signal.** The model has to infer phonetic content purely from the visual motion of lips, teeth, and tongue — many phonemes look nearly identical on the mouth (visemes), making this a genuinely ambiguous, information-poor input compared to audio speech recognition.
- **Variable-length sequences.** Spoken sentences vary in duration and are not evenly aligned to video frames, so the model needs a mechanism (CTC) to learn the alignment between frames and characters without explicit per-frame labels.
- **Spatiotemporal modeling.** The network must jointly capture *spatial* structure (mouth shape) and *temporal* structure (how that shape evolves), which is why this uses 3D convolutions rather than standard 2D CNNs.

## Architecture

The pipeline consists of the following stages:

1. **Video preprocessing** — Each input video is decoded frame by frame, converted to grayscale, and cropped to a fixed mouth-centered region of interest, producing a normalized tensor of shape `(frames, height, width, channels)`.
2. **Spatiotemporal feature extraction (Conv3D)** — A stack of 3D convolutional layers with max-pooling extracts features that span both space and time simultaneously, allowing the network to learn motion patterns of the mouth across consecutive frames rather than static shapes in isolated frames.
3. **Sequence modeling (Bidirectional LSTM)** — The extracted spatiotemporal features are flattened per timestep and passed into stacked Bidirectional LSTM layers, which model the sequence of mouth movements using both past and future context at every timestep.
4. **Dense projection** — A final `TimeDistributed(Dense)` layer projects each timestep's LSTM output onto a probability distribution over the character vocabulary.
5. **CTC loss** — Because there is no explicit frame-to-character alignment in the training data, the network is trained with Connectionist Temporal Classification loss, which marginalizes over all valid alignments between the input sequence and the target sentence, allowing the model to learn the alignment implicitly.
6. **Decoding** — At inference time, CTC greedy decoding (`tf.keras.backend.ctc_decode`) collapses the frame-level character predictions into the final predicted sentence.
