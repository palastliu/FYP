import re
import pandas as pd
from pathlib import Path

"""Remove URLs from text"""
def remove_urls(text: str) -> str: 
    return re.sub(r"http\S+|www\S+|https\S+", "", text)

"""Remove Twitter-style user mentions"""
def remove_mentions(text: str) -> str:

    return re.sub(r"@\w+", "", text)

"""Replace multiple spaces with a single space"""
def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

"""Light text cleaning for BERT input"""
def clean_text(text: str) -> str: 
    if not isinstance(text, str):
        return ""

    text = remove_urls(text)
    text = remove_mentions(text)
    text = normalize_whitespace(text)
    return text

"""Apply light preprocessing to the text column"""
def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].apply(clean_text)
    return df

"""Save the clean data to the dir."""
def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save DataFrame to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)