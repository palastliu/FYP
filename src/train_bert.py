import numpy as np
import pandas as pd
from pathlib import Path

from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.config import (
    TRAIN_CLEAN_FILE,
    VAL_CLEAN_FILE,
    MODEL_NAME,
    NUM_LABELS,
    MAX_LEN,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    BERT_MODEL_DIR,
)


def load_data(train_path, val_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    return train_df, val_df


def prepare_hf_dataset(df):
    df = df[["text", "label_id"]].copy()
    df = df.rename(columns={"label_id": "label"})
    return Dataset.from_pandas(df)


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_bert():
    train_df, val_df = load_data(TRAIN_CLEAN_FILE, VAL_CLEAN_FILE)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = prepare_hf_dataset(train_df)
    val_dataset = prepare_hf_dataset(val_df)

    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    BERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(BERT_MODEL_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_dir=str(BERT_MODEL_DIR / "logs"),
        logging_steps=100,
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(str(BERT_MODEL_DIR))
    tokenizer.save_pretrained(str(BERT_MODEL_DIR))

    eval_results = trainer.evaluate()
    print("\nValidation results:")
    print(eval_results)


if __name__ == "__main__":
    train_bert()