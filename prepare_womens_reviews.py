import pandas as pd
from pathlib import Path


def map_rating_to_label(rating):
    if pd.isna(rating):
        return None

    rating = int(rating)

    if rating in [1, 2]:
        return "negative"
    elif rating == 3:
        return "neutral"
    elif rating in [4, 5]:
        return "positive"
    return None


def main():
    input_file = Path(r"I:\MasterFyp\NLP_HYBRID_SENTIMENTAL\Womens Clothing E-Commerce Reviews.csv")
    output_file = Path(r"I:\MasterFyp\NLP_HYBRID_SENTIMENTAL\womens_reviews_prepared.csv")
    sample_file = Path(r"I:\MasterFyp\NLP_HYBRID_SENTIMENTAL\womens_reviews_5000.csv")

    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return

    print("Reading dataset...")
    df = pd.read_csv(input_file)

    required_cols = ["Unnamed: 0", "Review Text", "Rating"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Missing required column: {col}")
            return

    print("Preparing dataset...")
    prepared_df = df.copy()

    prepared_df = prepared_df.rename(columns={
        "Unnamed: 0": "review_id",
        "Review Text": "comment",
        "Rating": "score",
        "Title": "title",
        "Department Name": "department",
        "Class Name": "class_name"
    })

    # 去掉空评论
    prepared_df = prepared_df.dropna(subset=["comment", "score"]).copy()
    prepared_df["comment"] = prepared_df["comment"].astype(str).str.strip()
    prepared_df = prepared_df[prepared_df["comment"] != ""].copy()

    # 映射参考标签
    prepared_df["reference_label"] = prepared_df["score"].apply(map_rating_to_label)
    prepared_df = prepared_df.dropna(subset=["reference_label"]).copy()

    # 只保留主要列
    keep_cols = [
        "review_id",
        "comment",
        "score",
        "reference_label",
        "title",
        "department",
        "class_name"
    ]
    prepared_df = prepared_df[keep_cols]

    # 保存全量
    prepared_df.to_csv(output_file, index=False)

    # 抽样 5000 条做测试
    sample_df = prepared_df.sample(n=min(5000, len(prepared_df)), random_state=42)
    sample_df.to_csv(sample_file, index=False)

    print("\nDone.")
    print(f"Full prepared dataset saved to: {output_file}")
    print(f"Sample dataset saved to: {sample_file}")

    print("\nSample preview:")
    print(sample_df.head())

    print("\nReference label distribution in sample:")
    print(sample_df["reference_label"].value_counts())


if __name__ == "__main__":
    main()