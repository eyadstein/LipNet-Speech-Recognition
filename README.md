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

## Dataset

The model is trained and evaluated on the [GRID corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/), a widely used audiovisual sentence corpus consisting of short, fixed-grammar utterances (e.g. "bin blue at f two now") recorded from multiple speakers under controlled conditions. Its constrained grammar and vocabulary make it a standard benchmark for lipreading research.

## Tech stack

| Component | Tool |
|---|---|
| Language | Python 3.8+ |
| Deep learning framework | TensorFlow / Keras |
| Video & image processing | OpenCV |
| Numerical computing | NumPy |
| Visualization | Matplotlib, imageio |
| Dataset/checkpoint retrieval | gdown |

## Getting started

### Prerequisites

- Python 3.8 or later
- A CUDA-capable GPU is strongly recommended — training the Conv3D stack on CPU is impractically slow

### Installation

```bash
pip install opencv-python matplotlib imageio gdown tensorflow
```

### Usage

Open `LipNet.ipynb` and run the cells sequentially. The notebook is organized into the following sections:

| Section | Description |
|---|---|
| 0. Install and Import Dependencies | Environment setup and GPU availability check |
| 1. Build Data Loading Functions | Loads raw video files and alignment transcripts, builds the character-level vocabulary and lookup tables |
| 2. Create Data Pipeline | Constructs a `tf.data` pipeline with shuffling, padded batching, and prefetching for efficient training |
| 3. Design the Deep Neural Network | Defines the Conv3D + Bidirectional LSTM architecture |
| 4. Setup Training Options and Train | Compiles the model with CTC loss, sets up checkpointing and a learning-rate schedule, and trains for 100 epochs |
| 5. Make a Prediction | Loads trained weights and runs inference on held-out test videos |

Dataset and pretrained checkpoint downloads are handled automatically within the notebook via `gdown`.

## Training details

- **Loss function:** Connectionist Temporal Classification (CTC)
- **Optimizer:** Adam, initial learning rate `1e-4`
- **Learning-rate schedule:** Constant for the first 30 epochs, then exponential decay
- **Epochs:** 100
- **Checkpointing:** Model weights are saved after each epoch based on training loss
- **Monitoring:** A custom callback prints a sample prediction against the ground truth at the end of every epoch, providing a qualitative view of training progress

## Results

At inference, the model takes a raw video as input and outputs a predicted sentence, decoded via greedy CTC decoding. Example predictions and ground-truth comparisons are generated directly in the notebook's evaluation section.

## Limitations & future work

- Trained and evaluated on the GRID corpus, which has a small, fixed vocabulary and grammar — generalization to unconstrained, natural speech would require training on a larger and more diverse dataset (e.g. LRS2/LRS3).
- Greedy CTC decoding is used for simplicity; beam search decoding with a language model could improve accuracy on ambiguous predictions.
- No face/mouth detection and alignment step is included — the current pipeline assumes a fixed crop region, which would need to be replaced with a proper face-landmark-based crop for use on arbitrary, unconstrained video.

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for the full text.
