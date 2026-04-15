import pandas as pd
from pathlib import Path


REQUIRED_COLUMNS = ["id", "text", "label_id", "label", "split"]


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def check_required_columns(df: pd.DataFrame, required_columns=None) -> None:
    """Check whether the required columns exist in the DataFrame."""
    if required_columns is None:
        required_columns = REQUIRED_COLUMNS

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def inspect_dataset(df: pd.DataFrame, name: str = "dataset") -> None:
    """Print basic information about the dataset."""
    print(f"\n===== {name.upper()} =====")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nNull values:")
    print(df.isnull().sum())

    if "label" in df.columns:
        print("\nLabel distribution:")
        print(df["label"].value_counts())

    if "split" in df.columns:
        print("\nSplit distribution:")
        print(df["split"].value_counts())


def remove_null_rows(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Remove rows where the text column is null."""
    return df.dropna(subset=[text_col]).copy()


def remove_duplicate_texts(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Remove duplicate text rows."""
    return df.drop_duplicates(subset=[text_col]).copy()