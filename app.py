
import os
import sys

import gradio as gr

from src import config
from src.predict import SentimentPredictor

EMPTY_SUMMARY = "No reviews tested yet."


def summarize(history):
    by_movie = {}
    for movie, _, _, _, p_pos in history:
        by_movie.setdefault(movie, []).append(p_pos)
    if not by_movie:
        return EMPTY_SUMMARY
    lines = ["### Overall sentiment per movie"]
    for movie, probs in by_movie.items():
        avg = sum(probs) / len(probs)
        mood = "Positive" if avg >= 0.5 else "Negative"
        lines.append(f"- **{movie}**: {mood} ({avg * 100:.0f}% positive across {len(probs)} review(s))")
    return "\n".join(lines)


def build_demo(predictor):
    def analyze(movie, review, history):
        movie = (movie or "").strip() or "Untitled movie"
        review = (review or "").strip()
        if not review:
            raise gr.Error("Please write a review first.")
        p_pos = predictor.positive_probability(review)
        verdict = "Positive" if p_pos >= 0.5 else "Negative"
        confidence = max(p_pos, 1 - p_pos)
        short = review if len(review) <= 80 else review[:77] + "..."
        history = history + [[movie, short, verdict, f"{confidence * 100:.0f}%", p_pos]]
        table = [row[:4] for row in history]
        return {"Positive": p_pos, "Negative": 1 - p_pos}, table, summarize(history), history

    def clear():
        return None, [], EMPTY_SUMMARY, []

    with gr.Blocks(title="Movie Review Sentiment (DistilBERT + LoRA)") as demo:
        gr.Markdown(
            "# Movie Review Sentiment\n"
            "DistilBERT fine-tuned with LoRA (about 1% of parameters trained). "
            "The model reads the **review text**; the movie name is used to group your results."
        )
        history_state = gr.State([])

        with gr.Row():
            with gr.Column():
                movie_in = gr.Textbox(label="Movie name", placeholder="e.g. Inception")
                review_in = gr.Textbox(lines=6, label="Review", placeholder="Write a review of this movie...")
                with gr.Row():
                    go = gr.Button("Analyze", variant="primary")
                    reset = gr.Button("Clear history")
                gr.Examples(
                    examples=[
                        ["Inception", "A brilliant, mind-bending film. The score and visuals are unforgettable."],
                        ["Inception", "Too confusing, I lost interest halfway through."],
                        ["Cats", "Terrible acting and a boring plot. A complete waste of time."],
                        ["Parasite", "Not bad, but I expected more from all the hype."],
                    ],
                    inputs=[movie_in, review_in],
                )
            with gr.Column():
                label_out = gr.Label(num_top_classes=2, label="Prediction")
                summary_out = gr.Markdown(EMPTY_SUMMARY)

        table_out = gr.Dataframe(
            headers=["Movie", "Review", "Verdict", "Confidence"],
            label="Reviews tested",
            interactive=False,
        )

        inputs = [movie_in, review_in, history_state]
        outputs = [label_out, table_out, summary_out, history_state]
        go.click(analyze, inputs, outputs)
        review_in.submit(analyze, inputs, outputs)
        reset.click(clear, None, outputs)

    return demo


if __name__ == "__main__":
    if not os.path.isdir(config.ADAPTER_DIR):
        sys.exit(f"Adapter folder {config.ADAPTER_DIR} not found. Run: python train.py")
    build_demo(SentimentPredictor()).launch()  # launch(share=True) gives a public link
