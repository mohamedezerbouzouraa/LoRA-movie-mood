import torch
from transformers import AutoTokenizer

from src import config
from src.model import load_trained_model


class SentimentPredictor:
    def __init__(self, adapter_dir=config.ADAPTER_DIR):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.model = load_trained_model(adapter_dir).to(self.device)
        self.model.eval()

    def positive_probability(self, review):
        inputs = self.tokenizer(
            review, return_tensors="pt", truncation=True, max_length=config.MAX_LENGTH
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return float(torch.softmax(logits, dim=-1)[0][1])
