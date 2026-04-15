import pandas as pd
from pathlib import Path


def map_score_to_label(score):
    if pd.isna(score):
        return None

    score = int(score)

    if score in [1, 2]:
        return "negative"
    elif score == 3:
        return "neutral"
    elif score in [4, 5]:
        return "positive"
    return None


def main():
    input_file = Path(r"I:\MasterFyp\NLP_HYBRID_SENTIMENTAL\sentiment-analysis-dataset-google-play-app-reviews.csv")
    output_file = Path(r"I:\MasterFyp\NLP_HYBRID_SENTIMENTAL\google_play_reviews_1000.csv")
    full_output_file = Path(r"I:\MasterFyp\NLP_HYBRID_SENTIMENTAL\google_play_reviews_full_prepared.csv")

    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return

    print("Reading dataset...")
    df = pd.read_csv(input_file)

    required_cols = ["reviewId", "content", "score"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Missing required column: {col}")
            return

    print("Preparing dataset...")
    prepared_df = df[["reviewId", "content", "score"]].copy()
    prepared_df = prepared_df.rename(columns={
        "reviewId": "review_id",
        "content": "comment"
    })

    # remove empty comments
    prepared_df = prepared_df.dropna(subset=["comment", "score"])
    prepared_df["comment"] = prepared_df["comment"].astype(str).str.strip()
    prepared_df = prepared_df[prepared_df["comment"] != ""]

    # map score to reference label
    prepared_df["reference_label"] = prepared_df["score"].apply(map_score_to_label)

    # remove rows that failed mapping
    prepared_df = prepared_df.dropna(subset=["reference_label"])

    # save full prepared dataset
    prepared_df.to_csv(full_output_file, index=False)

    # sample 1000 rows for testing
    sample_df = prepared_df.sample(n=min(1000, len(prepared_df)), random_state=42)
    sample_df.to_csv(output_file, index=False)

    print("\nDone.")
    print(f"Full prepared dataset saved to: {full_output_file}")
    print(f"Sample dataset saved to: {output_file}")
    print("\nSample preview:")
    print(sample_df.head())

    print("\nReference label distribution in sample:")
    print(sample_df["reference_label"].value_counts())


if __name__ == "__main__":
    main()