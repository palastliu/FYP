import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.config import PREDICTIONS_DIR, TABLES_DIR, FIGURES_DIR, LABEL2ID


LABEL_ORDER = ["negative", "neutral", "positive"]


def load_prediction_file(file_path):
    return pd.read_csv(file_path)


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax)
    ax.set_title(title)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_models():
    bert_file = PREDICTIONS_DIR / "bert_test_outputs.csv"
    hybrid_file = PREDICTIONS_DIR / "hybrid_test_outputs.csv"

    print("Loading prediction files...")
    bert_df = load_prediction_file(bert_file)
    hybrid_df = load_prediction_file(hybrid_file)

    # Ground truth and predictions
    y_true = bert_df["label"]
    y_bert = bert_df["bert_pred"]
    y_hybrid = hybrid_df["hybrid_pred"]

    print("Computing metrics...")
    bert_metrics = compute_metrics(y_true, y_bert)
    hybrid_metrics = compute_metrics(y_true, y_hybrid)

    comparison_df = pd.DataFrame([
        {"model": "BERT", **bert_metrics},
        {"model": "BERT+Fuzzy", **hybrid_metrics},
    ])

    # Save metrics table
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = TABLES_DIR / "model_comparison.csv"
    comparison_df.to_csv(metrics_path, index=False)

    print("\nModel comparison:")
    print(comparison_df)

    # Save confusion matrices
    print("\nSaving confusion matrices...")
    plot_confusion_matrix(
        y_true, y_bert,
        labels=LABEL_ORDER,
        title="BERT Confusion Matrix",
        save_path=FIGURES_DIR / "bert_confusion_matrix.png"
    )

    plot_confusion_matrix(
        y_true, y_hybrid,
        labels=LABEL_ORDER,
        title="BERT+Fuzzy Confusion Matrix",
        save_path=FIGURES_DIR / "hybrid_confusion_matrix.png"
    )

    print("\nSaved files:")
    print(metrics_path)
    print(FIGURES_DIR / "bert_confusion_matrix.png")
    print(FIGURES_DIR / "hybrid_confusion_matrix.png")


if __name__ == "__main__":
    evaluate_models()