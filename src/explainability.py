import pandas as pd

from src.config import PREDICTIONS_DIR, TABLES_DIR


def load_hybrid_results(file_path):
    return pd.read_csv(file_path)


def generate_explanation(row):
    p_neg = row["p_neg"]
    p_neu = row["p_neu"]
    p_pos = row["p_pos"]
    hybrid_pred = row["hybrid_pred"]

    if hybrid_pred == "positive":
        return (
            f"The text is classified as positive because the positive score "
            f"({p_pos:.3f}) is stronger than the negative ({p_neg:.3f}) and "
            f"neutral ({p_neu:.3f}) scores under the fuzzy reasoning process."
        )
    elif hybrid_pred == "negative":
        return (
            f"The text is classified as negative because the negative score "
            f"({p_neg:.3f}) is stronger than the positive ({p_pos:.3f}) and "
            f"neutral ({p_neu:.3f}) scores under the fuzzy reasoning process."
        )
    else:
        return (
            f"The text is classified as neutral because the sentiment scores "
            f"show a more balanced or uncertain pattern "
            f"(neg={p_neg:.3f}, neu={p_neu:.3f}, pos={p_pos:.3f}), "
            f"which activates a neutral fuzzy decision."
        )


def select_case_studies(df, num_samples=15):
    # pick a mix of correct and incorrect cases
    correct_cases = df[df["label"] == df["hybrid_pred"]].copy()
    incorrect_cases = df[df["label"] != df["hybrid_pred"]].copy()

    correct_cases = correct_cases.head(num_samples // 2)
    incorrect_cases = incorrect_cases.head(num_samples - len(correct_cases))

    selected = pd.concat([correct_cases, incorrect_cases], axis=0).copy()
    return selected


def build_case_study_table(df):
    df = df.copy()
    df["explanation"] = df.apply(generate_explanation, axis=1)

    selected_columns = [
        "text",
        "label",
        "bert_pred",
        "hybrid_pred",
        "p_neg",
        "p_neu",
        "p_pos",
        "fuzzy_score",
        "explanation",
    ]
    return df[selected_columns]


def save_case_study_table(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    input_file = PREDICTIONS_DIR / "hybrid_test_outputs.csv"
    output_file = TABLES_DIR / "case_studies.csv"

    print("Loading hybrid results...")
    df = load_hybrid_results(input_file)

    print("Selecting case studies...")
    selected_df = select_case_studies(df, num_samples=15)

    print("Generating explanations...")
    case_table = build_case_study_table(selected_df)

    save_case_study_table(case_table, output_file)

    print("\nCase study file saved:")
    print(output_file)

    print("\nSample case studies:")
    print(case_table.head())


if __name__ == "__main__":
    main()