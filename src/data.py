from datasets import load_dataset

from src import config


def load_subsets():
    dataset = load_dataset(config.DATASET_NAME)
    print(f"Full dataset -> train: {len(dataset['train'])}, test: {len(dataset['test'])}")
    train_subset = dataset["train"].shuffle(seed=config.SEED).select(range(config.N_TRAIN))
    eval_subset = dataset["test"].shuffle(seed=config.SEED).select(range(config.N_EVAL))
    return train_subset, eval_subset


def tokenize_subsets(tokenizer, *subsets):

    def tokenize(batch):
        result = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_LENGTH,
        )
        result["labels"] = batch["label"]  # the Trainer expects the column to be named "labels"
        return result

    out = []
    for subset in subsets:
        subset = subset.map(tokenize, batched=True)
        subset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        out.append(subset)
    return out
