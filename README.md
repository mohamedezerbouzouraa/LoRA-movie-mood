# DistilBERT + LoRA Sentiment Classifier

My first project using LoRA, i've been studying and going deep into LoRA concepts and how it exactly works. 
You can see below everything about the project 

Fine-tune [DistilBERT](https://huggingface.co/distilbert-base-uncased) on the IMDb movie-review dataset using **LoRA** (Low-Rank Adaptation), then test it in a small web page.

LoRA freezes the pretrained model and trains tiny add-on matrices instead, so only about **1% of the parameters** are trained:

| | Trainable parameters |
|---|---|
| Full fine-tuning | ~67M (100%) |
| LoRA (r=8, `q_lin` + `v_lin`) | ~0.7M (~1%) |

## Project structure

```
distilbert-lora-sentiment/
├── src/
│   ├── config.py      # all settings (model, data size, LoRA and training values)
│   ├── data.py        # load and tokenize IMDb
│   ├── model.py       # build the model, attach LoRA, load a saved adapter
│   ├── metrics.py     # accuracy metric
│   └── predict.py     # score a single review with the trained model
├── train.py           # training entry point
├── app.py             # Gradio web page for testing reviews
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train (saves a small adapter to `./imdb-lora-adapter`):

```bash
python train.py
```

Open the test page:

```bash
python app.py
```

Type a movie name and a review, and the page shows Positive/Negative with confidence, keeps a table of everything you tested, and averages the sentiment per movie.

> The model reads only the review text. The movie name is used to group your results.

## Settings

Everything lives in `src/config.py`. Useful ones to change:

- `N_TRAIN` / `EPOCHS`: more data and epochs give higher accuracy (the defaults are a quick demo).
- `MAX_LENGTH`: 256 is faster, 512 is the DistilBERT maximum.
- `LORA_R` and `LORA_TARGET_MODULES`: try `r` = 4, 8, 16, or add `k_lin` / `out_lin` to see how accuracy and trainable parameters change.

Training on a CPU works but is slow. With an NVIDIA GPU, install the CUDA build of PyTorch first.

## Results

Fill in after your run:

| Setup | Trainable params | Test accuracy | Adapter size |
|---|---|---|---|
| LoRA r=8, 2,000 train reviews, 1 epoch | ~0.7M | _?_ | _?_ MB |
