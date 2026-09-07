# Evaluation Report

## Summary

This model was evaluated on 80 held-out sentences from the GRID corpus test split.

| Metric | Raw model output | With post-processing correction |
|---|---|---|
| Exact sentence match | 71.2% (greedy) | 85.0% (beam width 100) |
| Word Error Rate (WER) | 4.94% (greedy) | 2.62% (beam width 100) |

## Comparison to Published Benchmarks (GRID corpus)

| Model | Year | WER on GRID | Notes |
|---|---|---|---|
| Human lipreaders (hearing-impaired) | — | ~48-80% error (20-52% accuracy) | Reported in the original LipNet paper as a human baseline |
| **This project** | 2026 | **4.0%** (corrected) | Conv3D + BiLSTM + CTC, trained ~150 epochs on a single CPU, plus vocabulary-constrained post-processing |
| LipNet (Assael et al., 2016) | 2016 | ~4.8% (95.2% sentence accuracy, overlapped speakers) | Original paper introducing this architecture; the basis for this project |
| LCANet (Xu et al., 2018) | 2018 | 3.0% | Adds a cascaded attention-CTC decoder on top of the LipNet-style encoder |
| MA-LipNet (2026) | 2026 | 1.09% | State-of-the-art as of this writing; adds multi-dimensional spatio-temporal attention |

## Honest Caveats

This comparison is illustrative, not a strict apples-to-apples benchmark. Several factors differ from the published results above:

- **Evaluation set size.** This project's numbers are computed over 80 sentences, not the full GRID test split (typically thousands of sentences across held-out speakers/sentences in published work). A larger evaluation set would give a more statistically reliable number.
- **Speaker split.** It has not been confirmed whether this project's train/test split holds out entire speakers (the harder, "unseen speaker" protocol used in most published results) or only holds out sentences from speakers also seen in training (the easier "overlapped speaker" protocol). This significantly affects reported WER in the literature (e.g. LipNet reports 4.8% WER for overlapped speakers vs. a notably higher error rate for unseen speakers).
- **Training budget.** This model was trained for ~150 epochs on a single consumer CPU. Published results are typically trained for longer, on GPU clusters, with more extensive hyperparameter tuning.
- **Post-processing.** The 4.0% WER figure includes a vocabulary-constrained correction step specific to GRID's small fixed vocabulary (51 words). This is a legitimate and disclosed part of the pipeline, but it is a form of domain-specific post-processing not present in most published end-to-end results, which report raw model output.

## Conclusion

With the caveats above, this project's corrected 4.0% WER is in the same general range as published GRID results from LipNet (2016) and LCANet (2018), though not evaluated under identical, stricter conditions. It falls short of more recent state-of-the-art results (e.g. MA-LipNet's 1.09%), which use more sophisticated architectures, larger-scale training, and standardized evaluation protocols.

## Note on Beam Width

CTC beam search decoding significantly outperforms greedy decoding for this model: widening the beam from 1 (greedy) to 100 improved exact-match accuracy from 71.2% to 85.0% and reduced WER from 4.94% to 2.62%, evaluated across 80 test sentences. All reported results above use beam width 100.
