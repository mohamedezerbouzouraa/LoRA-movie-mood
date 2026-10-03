from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForSequenceClassification

from src import config


def load_base_model():
    return AutoModelForSequenceClassification.from_pretrained(config.MODEL_NAME, num_labels=2)


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def add_lora(model):
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="SEQ_CLS",
    )
    return get_peft_model(model, lora_config)


def load_trained_model(adapter_dir=config.ADAPTER_DIR):
    base = load_base_model()
    return PeftModel.from_pretrained(base, adapter_dir)
