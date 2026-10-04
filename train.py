import os

from transformers import AutoTokenizer, Trainer, TrainingArguments

from src import config
from src.data import load_subsets, tokenize_subsets
from src.metrics import compute_metrics
from src.model import add_lora, count_parameters, load_base_model


def main():
    train_raw, eval_raw = load_subsets()
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    train_ds, eval_ds = tokenize_subsets(tokenizer, train_raw, eval_raw)

    model = load_base_model()
    trainable, total = count_parameters(model)
    print(f"Baseline -> trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    lora_model = add_lora(model)
    lora_model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=config.RESULTS_DIR,
        eval_strategy="epoch",
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        logging_steps=50,
        save_steps=2000,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        report_to="none",
    )
    trainer = Trainer(
        model=lora_model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    print("Starting training with LoRA adapters...")
    trainer.train()

    results = trainer.evaluate()
    print(f"\nFinal accuracy on {config.N_EVAL} test reviews: {results['eval_accuracy']:.3f}")

    lora_model.save_pretrained(config.ADAPTER_DIR)
    size_mb = sum(
        os.path.getsize(os.path.join(config.ADAPTER_DIR, f))
        for f in os.listdir(config.ADAPTER_DIR)
    ) / 1e6
    print(f"Adapter saved to {config.ADAPTER_DIR} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
