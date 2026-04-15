from pathlib import Path

from src.config import TRAIN_FILE, VAL_FILE, TEST_FILE, PROCESSED_DATA_DIR
from src.data_loader import load_csv, check_required_columns, inspect_dataset
from src.preprocessing import preprocess_dataframe, save_dataframe


def main():
    train_df = load_csv(TRAIN_FILE)
    val_df = load_csv(VAL_FILE)
    test_df = load_csv(TEST_FILE)

    check_required_columns(train_df)
    check_required_columns(val_df)
    check_required_columns(test_df)

    print("\nBefore preprocessing:")
    inspect_dataset(train_df, "train")

    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)
    test_df = preprocess_dataframe(test_df)

    print("\nAfter preprocessing:")
    inspect_dataset(train_df, "train")

    print("\nSample cleaned texts:")
    print(train_df[["text", "label"]].head())

    # Save cleaned datasets
    save_dataframe(train_df, PROCESSED_DATA_DIR / "train_clean.csv")
    save_dataframe(val_df, PROCESSED_DATA_DIR / "val_clean.csv")
    save_dataframe(test_df, PROCESSED_DATA_DIR / "test_clean.csv")

    print("\nCleaned files saved successfully.")
    print(PROCESSED_DATA_DIR / "train_clean.csv")
    print(PROCESSED_DATA_DIR / "val_clean.csv")
    print(PROCESSED_DATA_DIR / "test_clean.csv")


if __name__ == "__main__":
    main()