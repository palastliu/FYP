import pandas as pd


COMMENT_CANDIDATES = [
    "comment",
    "review",
    "review text",
    "review_text",
    "text",
    "content",
    "body",
    "comment text",
    "comment_text",
]

ID_CANDIDATES = [
    "review_id",
    "id",
    "unnamed: 0",
    "index",
]

SCORE_CANDIDATES = [
    "score",
    "rating",
    "stars",
    "star",
]


def normalize_colname(col_name: str) -> str:
    return str(col_name).strip().lower().replace("_", " ")


def guess_column(columns, candidates):
    normalized_map = {normalize_colname(col): col for col in columns}
    for candidate in candidates:
        if candidate in normalized_map:
            return normalized_map[candidate]
    return None


def guess_comment_column(columns):
    return guess_column(columns, COMMENT_CANDIDATES)


def guess_id_column(columns):
    return guess_column(columns, ID_CANDIDATES)


def guess_score_column(columns):
    return guess_column(columns, SCORE_CANDIDATES)


def map_rating_to_label(score):
    if pd.isna(score):
        return None

    try:
        score = float(score)
    except Exception:
        return None

    # 通用 5 分制映射
    if score in [1, 2]:
        return "negative"
    elif score == 3:
        return "neutral"
    elif score in [4, 5]:
        return "positive"
    else:
        return None


def preprocess_uploaded_csv(
    raw_df: pd.DataFrame,
    comment_col: str,
    id_col: str = None,
    score_col: str = None,
):
    """
    Convert an arbitrary review CSV into the internal standard format:
    - review_id
    - comment
    - optional score
    - optional reference_label
    """

    if comment_col is None or comment_col not in raw_df.columns:
        raise ValueError("A valid comment column must be selected.")

    df = raw_df.copy()

    # comment
    df["comment"] = df[comment_col].astype(str).str.strip()
    df = df[df["comment"] != ""].copy()

    # review_id
    if id_col and id_col in df.columns:
        df["review_id"] = df[id_col].astype(str)
    else:
        df["review_id"] = range(1, len(df) + 1)

    output_cols = ["review_id", "comment"]

    # score + reference_label
    if score_col and score_col in df.columns:
        df["score"] = df[score_col]
        df["reference_label"] = df["score"].apply(map_rating_to_label)
        output_cols.extend(["score", "reference_label"])

    processed_df = df[output_cols].copy()

    return processed_df