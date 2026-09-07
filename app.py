import gradio as gr
from predict import predict, CHECKPOINT_PATH
import os


def run_prediction(video_file):
    if video_file is None:
        return "No video uploaded.", "No video uploaded."

    if not os.path.exists(CHECKPOINT_PATH):
        return (
            "Model checkpoint not found.",
            "Train the model first or place checkpoint.weights.h5 in models/."
        )

    try:
        raw_pred, corrected_pred = predict(video_file)
        return raw_pred, corrected_pred
    except Exception as e:
        return f"Error: {e}", f"Error: {e}"


with gr.Blocks(title="LipNet — Sentence-Level Lipreading") as demo:
    gr.Markdown(
        """
        # LipNet — Sentence-Level Lipreading
        Upload a **GRID-format video** (a person facing the camera, saying a
        short fixed-grammar sentence) and the model will predict the spoken
        sentence from lip movement alone — no audio used.

        ⚠️ This model was trained exclusively on the GRID corpus's constrained
        vocabulary (51 words: commands, colors, prepositions, letters, digits,
        adverbs). It will **not** correctly transcribe free/natural speech or
        webcam video outside GRID-like conditions — see the README for details.
        """
    )

    with gr.Row():
        video_input = gr.Video(label="Upload GRID-format video (.mpg)")

    predict_btn = gr.Button("Predict", variant="primary")

    with gr.Row():
        raw_output = gr.Textbox(label="Raw CTC Prediction")
        corrected_output = gr.Textbox(label="Corrected Prediction (GRID vocabulary)")

    predict_btn.click(
        fn=run_prediction,
        inputs=video_input,
        outputs=[raw_output, corrected_output]
    )

    gr.Markdown(
        """
        ---
        Try a sample video from `data/s1/` in this repository, e.g. `bbal6n.mpg`.
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
