import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from transformers import BertTokenizer, BertForSequenceClassification

from src.config import (
    VAL_CLEAN_FILE,
    TEST_CLEAN_FILE,
    BERT_MODEL_DIR,
    MAX_LEN,
    ID2LABEL,
    PREDICTIONS_DIR,
)


def load_data(file_path):
    return pd.read_csv(file_path)


def load_model_and_tokenizer(model_dir):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, device


def predict_probabilities(df, model, tokenizer, device, batch_size=32):
    texts = df["text"].tolist()
    all_probs = []
    all_preds = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits.cpu().numpy()
            probs = softmax(logits, axis=1)
            preds = np.argmax(probs, axis=1)

        all_probs.extend(probs)
        all_preds.extend(preds)

    return np.array(all_probs), np.array(all_preds)


def build_output_dataframe(df, probs, preds):
    result_df = df.copy()
    result_df["p_neg"] = probs[:, 0]
    result_df["p_neu"] = probs[:, 1]
    result_df["p_pos"] = probs[:, 2]
    result_df["bert_pred_id"] = preds
    result_df["bert_pred"] = result_df["bert_pred_id"].map(ID2LABEL)
    return result_df


def save_predictions(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def run_single_inference(input_file, output_file, model, tokenizer, device, split_name):
    print(f"\nRunning inference on {split_name} set...")
    df = load_data(input_file)

    probs, preds = predict_probabilities(df, model, tokenizer, device)
    result_df = build_output_dataframe(df, probs, preds)

    save_predictions(result_df, output_file)

    print(f"Saved file: {output_file}")
    print(result_df[["text", "label", "p_neg", "p_neu", "p_pos", "bert_pred"]].head())


def run_inference():
    print("Loading trained BERT model...")
    model, tokenizer, device = load_model_and_tokenizer(BERT_MODEL_DIR)
    print(f"Using device: {device}")

    val_output_file = PREDICTIONS_DIR / "bert_val_outputs.csv"
    test_output_file = PREDICTIONS_DIR / "bert_test_outputs.csv"

    run_single_inference(
        input_file=VAL_CLEAN_FILE,
        output_file=val_output_file,
        model=model,
        tokenizer=tokenizer,
        device=device,
        split_name="validation"
    )

    run_single_inference(
        input_file=TEST_CLEAN_FILE,
        output_file=test_output_file,
        model=model,
        tokenizer=tokenizer,
        device=device,
        split_name="test"
    )

    print("\nInference for validation and test sets completed.")


if __name__ == "__main__":
    run_inference()