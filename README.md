# LipNet — Sentence-Level Lipreading from Silent Video

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)

## Overview

LipNet is a deep learning system that performs **sentence-level lipreading**: given a silent video of a person speaking, it predicts the full sentence being spoken, without any audio input. This repository implements a Conv3D + Bidirectional LSTM architecture trained end-to-end with Connectionist Temporal Classification (CTC) loss to map variable-length video sequences directly to sentences.

Rather than classifying isolated words or relying on hand-crafted visual features and separate language models, the network learns spatiotemporal visual features and character-level sequence alignment jointly, in a single trainable pipeline.

## Results

Evaluated on 80 held-out test sentences from the GRID corpus:

| Metric | Raw model output | With post-processing correction |
|---|---|---|
| Exact sentence match | 35.0% | **76.2%** |
| Word Error Rate (WER) | 15.0% | **4.0%** |

For reference, the original LipNet paper reports ~4.8% WER on the GRID corpus with a substantially larger training run. This result was achieved training on a single consumer CPU.

### Why raw vs. corrected?

CTC decoding collapses consecutive repeated characters unless the model explicitly emits a blank token between them. In practice, this meant the model consistently mangled words containing double letters — `green` -> `gren`, `three` -> `thre`, `soon` -> `son` — while getting essentially everything else right.

Since GRID sentences are built from a small, fixed vocabulary (4 commands, 4 colors, 4 prepositions, letters, digits, 4 adverbs — 51 words total), a lightweight nearest-match correction against that vocabulary resolves this decoding artifact without any additional training, taking exact-match accuracy from 35% to 76%.

## Why this problem is hard

- **No audio signal.** The model has to infer phonetic content purely from the visual motion of lips, teeth, and tongue — many phonemes look nearly identical on the mouth (visemes), making this a genuinely ambiguous, information-poor input compared to audio speech recognition.
- **Variable-length sequences.** Spoken sentences vary in duration and are not evenly aligned to video frames, so the model needs a mechanism (CTC) to learn the alignment between frames and characters without explicit per-frame labels.
- **Spatiotemporal modeling.** The network must jointly capture spatial structure (mouth shape) and temporal structure (how that shape evolves), which is why this uses 3D convolutions rather than standard 2D CNNs.

## Architecture

The pipeline consists of the following stages:

1. **Video preprocessing** — Each input video is decoded frame by frame, converted to grayscale, and cropped to a fixed mouth-centered region of interest, producing a normalized tensor of shape `(frames, height, width, channels)`.
2. **Spatiotemporal feature extraction (Conv3D)** — A stack of 3D convolutional layers with max-pooling extracts features that span both space and time simultaneously.
3. **Sequence modeling (Bidirectional LSTM)** — The extracted spatiotemporal features are flattened per timestep and passed into stacked Bidirectional LSTM layers, modeling the sequence of mouth movements using both past and future context at every timestep.
4. **Dense projection** — A final `TimeDistributed(Dense)` layer projects each timestep's LSTM output onto a probability distribution over the character vocabulary.
5. **CTC loss** — The network is trained with Connectionist Temporal Classification loss, which marginalizes over all valid alignments between the input sequence and the target sentence, letting the model learn the alignment implicitly.
6. **Decoding + correction** — CTC beam search decoding converts frame-level character predictions into text, followed by a vocabulary-constrained correction step that resolves double-letter decoding artifacts (see Results above).

Video -> Grayscale + Mouth Crop -> Conv3D x3 -> Bi-LSTM x2 -> Dense (softmax) -> CTC Decode -> Vocab Correction -> Sentence

## Dataset

The model is trained and evaluated on the [GRID corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/), a widely used audiovisual sentence corpus consisting of short, fixed-grammar utterances (e.g. "bin blue at f two now") recorded from multiple speakers under controlled conditions. Its constrained grammar and vocabulary make it a standard benchmark for lipreading research.

## Tech stack

| Component | Tool |
|---|---|
| Language | Python 3.11 |
| Deep learning framework | TensorFlow / Keras 3 |
| Video & image processing | OpenCV |
| Numerical computing | NumPy |
| Visualization | Matplotlib, imageio |
| Dataset/checkpoint retrieval | gdown |

## Getting started

### Prerequisites

- Python 3.11 (TensorFlow does not yet support 3.13+)
- A CUDA-capable GPU is strongly recommended — training the Conv3D stack on CPU takes on the order of tens of hours

### Installation

python -m venv venv
venv\Scripts\activate
pip install opencv-python matplotlib imageio gdown tensorflow

### Usage

Open `LipNet.ipynb` and run the cells sequentially. The notebook is organized into the following sections:

| Section | Description |
|---|---|
| 0. Install and Import Dependencies | Environment setup and GPU availability check |
| 1. Paths and Data Setup | Downloads and locates the GRID corpus data |
| 2. Data Loading Functions | Loads videos + alignment transcripts, builds the character-level vocabulary |
| 3. Data Pipeline | Constructs a `tf.data` pipeline with shuffling, padded batching, and prefetching |
| 4. Model Architecture | Defines the Conv3D + Bidirectional LSTM network |
| 5. Training Setup | CTC loss, learning-rate schedule, checkpointing, and example-prediction callback |
| 6. Train | Resumable training loop with automatic checkpoint loading |
| 7. Predict on a Test Sample | Runs inference and applies vocabulary-based correction |

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for the full text.
