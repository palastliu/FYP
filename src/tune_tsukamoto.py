import itertools
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.config import PREDICTIONS_DIR
from src.fuzzy_system import get_default_params, infer_sentiment


def evaluate_params(df, params):
    preds = []

    for _, row in df.iterrows():
        _, label, _ = infer_sentiment(
            p_neg=row["p_neg"],
            p_neu=row["p_neu"],
            p_pos=row["p_pos"],
            params=params
        )
        preds.append(label)

    acc = accuracy_score(df["label"], preds)
    f1 = f1_score(df["label"], preds, average="weighted")
    return acc, f1


def build_params(low_a, low_b, med_a, med_b, med_c, high_a, high_b):
    params = get_default_params()

    for var in ["neg", "neu", "pos"]:
        params[var]["low"]["a"] = low_a
        params[var]["low"]["b"] = low_b

        params[var]["medium"]["a"] = med_a
        params[var]["medium"]["b"] = med_b
        params[var]["medium"]["c"] = med_c

        params[var]["high"]["a"] = high_a
        params[var]["high"]["b"] = high_b

    return params


def main():
    input_file = PREDICTIONS_DIR / "bert_val_outputs.csv"
    df = pd.read_csv(input_file)

    low_a_list = [0.25, 0.30, 0.35]
    low_b_list = [0.45, 0.48, 0.50]
    med_a_list = [0.20, 0.25, 0.30]
    med_b_list = [0.45, 0.48, 0.50]
    med_c_list = [0.75, 0.78, 0.80]
    high_a_list = [0.45, 0.48, 0.50]
    high_b_list = [0.68, 0.70, 0.75]

    results = []
    best_result = None

    total = 0

    for low_a, low_b, med_a, med_b, med_c, high_a, high_b in itertools.product(
        low_a_list, low_b_list, med_a_list, med_b_list, med_c_list, high_a_list, high_b_list
    ):
        # 基本合法性约束
        if not (low_a < low_b):
            continue
        if not (med_a < med_b < med_c):
            continue
        if not (high_a < high_b):
            continue

        # 让 membership 结构大致合理
        if not (low_a <= med_a <= med_b):
            continue
        if not (med_b <= high_b):
            continue

        total += 1

        params = build_params(low_a, low_b, med_a, med_b, med_c, high_a, high_b)
        acc, f1 = evaluate_params(df, params)

        row = {
            "low_a": low_a,
            "low_b": low_b,
            "med_a": med_a,
            "med_b": med_b,
            "med_c": med_c,
            "high_a": high_a,
            "high_b": high_b,
            "accuracy": acc,
            "f1_score": f1,
        }
        results.append(row)

        if best_result is None or (f1 > best_result["f1_score"]) or (
            f1 == best_result["f1_score"] and acc > best_result["accuracy"]
        ):
            best_result = row
            print("New best:", best_result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=["f1_score", "accuracy"], ascending=False)

    output_file = PREDICTIONS_DIR / "tsukamoto_tuning_results_val.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\nTotal valid parameter combinations tested: {total}")
    print("\nBest params found on validation set:")
    print(best_result)
    print(f"\nSaved tuning results to: {output_file}")


if __name__ == "__main__":
    main()