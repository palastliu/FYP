import pandas as pd

from src.config import PREDICTIONS_DIR
from src.fuzzy_system import infer_sentiment, get_default_params


def load_bert_outputs(file_path):
    return pd.read_csv(file_path)


def run_hybrid_inference(df, params=None):
    df = df.copy()

    if params is None:
        params = get_default_params()

    fuzzy_scores = []
    hybrid_preds = []
    top_rules = []

    for _, row in df.iterrows():
        score, label, fired_rules = infer_sentiment(
            p_neg=row["p_neg"],
            p_neu=row["p_neu"],
            p_pos=row["p_pos"],
            params=params
        )

        fuzzy_scores.append(score)
        hybrid_preds.append(label)

        if fired_rules:
            top_rule = max(fired_rules, key=lambda x: x["alpha"])["rule"]
        else:
            top_rule = "fallback_rule"

        top_rules.append(top_rule)

    df["fuzzy_score"] = fuzzy_scores
    df["hybrid_pred"] = hybrid_preds
    df["top_rule"] = top_rules

    return df


def save_hybrid_outputs(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    input_file = PREDICTIONS_DIR / "bert_test_outputs.csv"
    output_file = PREDICTIONS_DIR / "hybrid_test_outputs.csv"

    print("Loading BERT output file...")
    df = load_bert_outputs(input_file)

    print("Loading default Tsukamoto parameters...")
    params = get_default_params()

    print("Running hybrid inference...")
    result_df = run_hybrid_inference(df, params=params)

    save_hybrid_outputs(result_df, output_file)

    print("\nHybrid inference completed.")
    print(f"Saved file: {output_file}")

    print("\nSample outputs:")
    print(result_df[[
        "text", "label", "bert_pred", "p_neg", "p_neu", "p_pos",
        "fuzzy_score", "hybrid_pred", "top_rule"
    ]].head())


if __name__ == "__main__":
    main()