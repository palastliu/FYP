from collections import Counter, defaultdict
from pathlib import Path
import re

print("APP_BATCH_ANALYZER FINAL VERIFIED VERSION LOADED")

import matplotlib.pyplot as plt
import pandas as pd
import torch
import spacy
from scipy.special import softmax
from transformers import BertTokenizer, BertForSequenceClassification

from src.config import BERT_MODEL_DIR, MAX_LEN, OUTPUT_DIR
from src.fuzzy_system import infer_sentiment, get_default_params
from src.aspect_mapper import map_aspect, aspect_priority


try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    NLP = None


POSITIVE_CUES = {
    "good", "great", "excellent", "amazing", "awesome", "easy", "helpful",
    "smooth", "love", "like", "best", "nice", "useful", "fast", "stable",
    "recommend", "perfect", "happy", "well", "better", "clean", "simple"
}

NEGATIVE_CUES = {
    "bad", "poor", "slow", "terrible", "awful", "disappointing", "worst",
    "crash", "freeze", "drain", "bug", "issue", "problem", "broken",
    "late", "hard", "difficult", "annoying", "waste", "frustrating",
    "unhelpful", "fail", "error", "wrong", "expensive"
}

NEUTRAL_CUES = {
    "okay", "fine", "average", "normal", "standard", "available",
    "version", "option", "update", "feature"
}

NEGATIONS = {"not", "no", "never", "none", "hardly", "barely", "n't"}

PRONOUN_STARTS = {"it", "this", "that", "which", "these", "those"}
GENERIC_WEAK_WORDS = {
    "issue", "problem", "thing", "stuff", "way", "time", "app", "version"
}
WEAK_VERBS = {"be", "have", "do", "get", "make"}
BAD_EXACT_PHRASES = {
    "this is issue",
    "that is issue",
    "it is issue",
    "which is annoying",
    "this is bad",
    "that is bad",
    "it is bad"
}


def make_safe_name(name: str) -> str:
    name = Path(str(name)).stem.lower().strip()
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "dataset"


def build_output_paths(input_path: Path):
    dataset_name = make_safe_name(input_path.stem)
    report_dir = OUTPUT_DIR / "batch_outputs" / f"{dataset_name}_output"
    figures_dir = report_dir / "figures"
    excel_file = report_dir / "comment_analysis_report.xlsx"
    return dataset_name, report_dir, figures_dir, excel_file


def load_model_and_tokenizer(model_dir):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, device


def preprocess_text(text: str) -> str:
    return str(text).strip()


def predict_bert_probabilities(text, model, tokenizer, device):
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits.cpu().numpy()
        probs = softmax(logits, axis=1)[0]

    return probs[0], probs[1], probs[2]


def generate_risk_level(final_label):
    if final_label == "negative":
        return "High"
    elif final_label == "neutral":
        return "Medium"
    return "Low"


def generate_action_hint(final_label):
    if final_label == "negative":
        return "Immediate review recommended."
    elif final_label == "neutral":
        return "Monitor this feedback."
    return "Positive feedback. Keep as supporting evidence."


def analyze_comment(text, model, tokenizer, device, params):
    clean_text = preprocess_text(text)

    p_neg, p_neu, p_pos = predict_bert_probabilities(clean_text, model, tokenizer, device)

    final_score, final_label, fired_rules = infer_sentiment(
        p_neg=p_neg,
        p_neu=p_neu,
        p_pos=p_pos,
        params=params
    )

    if fired_rules:
        top_rule = max(fired_rules, key=lambda x: x["alpha"])["rule"]
    else:
        top_rule = "fallback_rule"

    return {
        "comment": clean_text,
        "final_label": final_label,
        "risk_level": generate_risk_level(final_label),
        "action_hint": generate_action_hint(final_label),
        "top_rule": top_rule,
        "score": final_score,
    }


def normalize_phrase(text):
    return " ".join(str(text).strip().lower().split())


def has_negation(token):
    for child in token.children:
        if child.lower_ in NEGATIONS or child.dep_ == "neg":
            return True
    return False


def flip_scores_if_negated(pos_score, neg_score, is_negated):
    if is_negated:
        return neg_score, pos_score
    return pos_score, neg_score


def score_phrase_text(phrase_text):
    words = phrase_text.lower().split()

    pos_score = 0.0
    neg_score = 0.0
    neu_score = 0.0

    for w in words:
        if w in POSITIVE_CUES:
            pos_score += 1.0
        if w in NEGATIVE_CUES:
            neg_score += 1.0
        if w in NEUTRAL_CUES:
            neu_score += 0.8

    if pos_score == 0 and neg_score == 0:
        neu_score += 0.6

    return pos_score, neg_score, neu_score


def assign_evidence_label(pos_score, neg_score, neu_score, min_margin=0.35):
    scores = {
        "positive": pos_score,
        "negative": neg_score,
        "neutral": neu_score
    }
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top_label, top_score = ranked[0]
    second_score = ranked[1][1]

    if top_score <= 0:
        return None, 0.0

    if (top_score - second_score) < min_margin:
        return None, top_score - second_score

    return top_label, top_score - second_score


def is_natural_phrase(phrase):
    if phrase in BAD_EXACT_PHRASES:
        return False

    doc = NLP(phrase)
    tokens = [t for t in doc if t.is_alpha]
    words = [t.text.lower() for t in tokens]

    if len(words) < 2:
        return False

    if len(words) >= 2 and len(set(words)) == 1:
        return False

    if words[0] in PRONOUN_STARTS:
        has_strong_noun = any(
            t.pos_ in {"NOUN", "PROPN"} and t.lemma_.lower() not in GENERIC_WEAK_WORDS
            for t in tokens[1:]
        )
        if not has_strong_noun:
            return False

    if len(words) == 2 and words[0] in {"problem", "issue"} and words[1] in {"app", "version"}:
        return False

    if len(words) == 2 and words[0] in WEAK_VERBS and words[1] in GENERIC_WEAK_WORDS:
        return False

    return True


def extract_evidence_candidates(text):
    if NLP is None:
        return []

    doc = NLP(str(text))
    candidates = []

    for token in doc:
        if token.lemma_ == "be" and token.pos_ in {"AUX", "VERB"}:
            subj = None
            comp = None

            for child in token.children:
                if child.dep_ in {"nsubj", "nsubjpass"} and child.pos_ in {"NOUN", "PROPN", "PRON"}:
                    subj = child
                if child.dep_ in {"acomp", "attr", "oprd"} and child.pos_ in {"ADJ", "NOUN", "PROPN"}:
                    comp = child

            if subj is not None and comp is not None:
                phrase = normalize_phrase(f"{subj.text} {token.text} {comp.text}")
                candidates.append((phrase, token))

    for token in doc:
        if token.pos_ == "VERB":
            adv = None
            dobj = None

            for child in token.children:
                if child.dep_ == "advmod" and child.pos_ == "ADV":
                    adv = child
                if child.dep_ in {"dobj", "obj"} and child.pos_ in {"NOUN", "PROPN"}:
                    dobj = child

            if adv is not None:
                phrase = normalize_phrase(f"{token.text} {adv.text}")
                candidates.append((phrase, token))

            if dobj is not None:
                phrase = normalize_phrase(f"{token.text} {dobj.text}")
                candidates.append((phrase, token))

    filtered_tokens = [t for t in doc if t.is_alpha and not t.is_stop]

    for i in range(len(filtered_tokens) - 1):
        t1 = filtered_tokens[i]
        t2 = filtered_tokens[i + 1]

        if t1.pos_ == "ADV" and t2.pos_ == "ADJ":
            phrase = normalize_phrase(f"{t1.text} {t2.text}")
            candidates.append((phrase, t2))

        elif t1.pos_ == "ADJ" and t2.pos_ in {"NOUN", "PROPN"}:
            phrase = normalize_phrase(f"{t1.text} {t2.text}")
            candidates.append((phrase, t2))

        elif t1.pos_ in {"NOUN", "PROPN"} and t2.pos_ in {"NOUN", "PROPN"}:
            phrase = normalize_phrase(f"{t1.text} {t2.text}")
            candidates.append((phrase, t2))

    seen = set()
    unique_candidates = []

    for phrase, anchor in candidates:
        phrase_doc = NLP(phrase)
        tokens = [tok for tok in phrase_doc if tok.is_alpha]
        has_noun = any(tok.pos_ in {"NOUN", "PROPN"} for tok in tokens)
        token_count = len(tokens)

        if phrase in seen:
            continue

        if not (token_count >= 2 and (token_count >= 3 or has_noun)):
            continue

        if not is_natural_phrase(phrase):
            continue

        seen.add(phrase)
        unique_candidates.append((phrase, anchor))

    return unique_candidates


def classify_evidence_phrases(text):
    if NLP is None:
        return []

    candidates = extract_evidence_candidates(text)
    evidence_rows = []

    for phrase, anchor in candidates:
        pos_score, neg_score, neu_score = score_phrase_text(phrase)

        is_negated = has_negation(anchor)
        pos_score, neg_score = flip_scores_if_negated(pos_score, neg_score, is_negated)

        label, strength = assign_evidence_label(pos_score, neg_score, neu_score, min_margin=0.35)
        if label is None:
            continue

        evidence_rows.append({
            "evidence_phrase": phrase,
            "evidence_label": label,
            "evidence_strength": round(strength, 3)
        })

    return evidence_rows


def build_evidence_summary(df, top_n=10):
    print("Building evidence summary with external aspect mapper...")

    evidence_map = defaultdict(lambda: defaultdict(Counter))
    strength_map = defaultdict(dict)

    for _, row in df.iterrows():
        text = row["comment"]
        row_label = row["final_label"]

        evidence_rows = classify_evidence_phrases(text)

        for evidence in evidence_rows:
            if evidence["evidence_label"] != row_label:
                continue

            aspect = map_aspect(evidence["evidence_phrase"])
            phrase = evidence["evidence_phrase"]

            evidence_map[row_label][aspect].update([phrase])

            key = (aspect, phrase)
            prev_strength = strength_map[row_label].get(key, 0.0)
            strength_map[row_label][key] = max(prev_strength, evidence["evidence_strength"])

    rows = []
    for label in ["negative", "neutral", "positive"]:
        temp_rows = []
        for aspect, counter in evidence_map[label].items():
            for phrase, freq in counter.items():
                temp_rows.append({
                    "label": label,
                    "aspect": aspect,
                    "evidence_phrase": phrase,
                    "frequency": freq,
                    "evidence_strength": strength_map[label][(aspect, phrase)]
                })

        temp_rows = sorted(
            temp_rows,
            key=lambda x: (
                aspect_priority(x["aspect"]),
                -x["evidence_strength"],
                -x["frequency"]
            )
        )

        rows.extend(temp_rows[:top_n])

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df[["label", "aspect", "evidence_phrase", "frequency"]]

    return result_df


def create_summary_sheet(result_df, original_df=None):
    label_counts = result_df["final_label"].value_counts()
    total = len(result_df)

    rows = []
    for label in ["negative", "neutral", "positive"]:
        count = int(label_counts.get(label, 0))
        proportion = count / total if total > 0 else 0

        rows.append({"metric": f"{label}_count", "value": count})
        rows.append({"metric": f"{label}_proportion", "value": proportion})

    rows.append({"metric": "total_comments", "value": total})

    if original_df is not None and "reference_label" in original_df.columns:
        comparison_df = result_df[["review_id", "final_label"]].merge(
            original_df[["review_id", "reference_label"]],
            on="review_id",
            how="left"
        )
        comparison_df["match"] = comparison_df.apply(
            lambda row: "yes" if row["final_label"] == row["reference_label"] else "no",
            axis=1
        )
        overall_match = (comparison_df["match"] == "yes").mean()
        rows.append({"metric": "overall_match_rate", "value": overall_match})

    return pd.DataFrame(rows)


def plot_label_distribution_pie(df, save_path):
    counts = df["final_label"].value_counts().reindex(
        ["negative", "neutral", "positive"],
        fill_value=0
    )

    plt.figure(figsize=(5, 5))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    plt.title("Comment Label Distribution")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_evidence_bar(evidence_df, label_name, save_path):
    label_df = evidence_df[evidence_df["label"] == label_name].head(10)
    if label_df.empty:
        return

    labels = [
        f"{row['aspect']}: {row['evidence_phrase']}"
        for _, row in label_df.iterrows()
    ]

    plt.figure(figsize=(10, 4.8))
    plt.bar(labels, label_df["frequency"])
    plt.title(f"Top Evidence Phrases - {label_name}")
    plt.xlabel("Aspect : Evidence Phrase")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_excel_report(summary_df, all_df, evidence_df, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        all_df.to_excel(writer, sheet_name="all_comments", index=False)
        evidence_df.to_excel(writer, sheet_name="evidence_summary", index=False)


def main():
    print("Loading trained BERT model...")
    model, tokenizer, device = load_model_and_tokenizer(BERT_MODEL_DIR)
    print(f"Using device: {device}")

    if NLP is None:
        print("spaCy model 'en_core_web_sm' is not installed.")
        print("Please run: python -m spacy download en_core_web_sm")
        return

    params = get_default_params()

    input_path = input("\nEnter input CSV path: ").strip()
    if not input_path:
        print("No file path provided.")
        return

    input_file = Path(input_path)
    if not input_file.exists():
        print(f"File not found: {input_file}")
        return

    dataset_name, report_dir, figures_dir, excel_file = build_output_paths(input_file)

    df = pd.read_csv(input_file)

    if "comment" not in df.columns:
        print("Input CSV must contain a 'comment' column.")
        return

    if "review_id" not in df.columns:
        df.insert(0, "review_id", range(1, len(df) + 1))

    print(f"\nAnalyzing {len(df)} comments...")
    print(f"Output folder: {report_dir}")

    results = []
    for _, row in df.iterrows():
        result = analyze_comment(
            text=row["comment"],
            model=model,
            tokenizer=tokenizer,
            device=device,
            params=params
        )
        result["review_id"] = row["review_id"]
        results.append(result)

    result_df = pd.DataFrame(results)

    comment_aspects = []
    for _, row in result_df.iterrows():
        evidence_rows = classify_evidence_phrases(row["comment"])
        matched = [e for e in evidence_rows if e["evidence_label"] == row["final_label"]]

        aspect = "general"
        if matched:
            enriched = []
            for e in matched:
                e_aspect = map_aspect(e["evidence_phrase"])
                enriched.append({
                    "aspect": e_aspect,
                    "strength": e["evidence_strength"]
                })

            enriched = sorted(
                enriched,
                key=lambda x: (
                    aspect_priority(x["aspect"]),
                    -x["strength"]
                )
            )
            aspect = enriched[0]["aspect"]

        comment_aspects.append(aspect)

    result_df["aspect"] = comment_aspects
    result_df = result_df[
        ["review_id", "comment", "final_label", "aspect", "risk_level", "action_hint", "top_rule"]
    ]

    evidence_summary_df = build_evidence_summary(result_df, top_n=10)
    summary_df = create_summary_sheet(result_df, original_df=df)

    figures_dir.mkdir(parents=True, exist_ok=True)

    save_excel_report(
        summary_df=summary_df,
        all_df=result_df,
        evidence_df=evidence_summary_df,
        output_file=excel_file
    )

    plot_label_distribution_pie(result_df, figures_dir / "label_distribution_pie.png")
    plot_evidence_bar(evidence_summary_df, "negative", figures_dir / "negative_keywords_bar.png")
    plot_evidence_bar(evidence_summary_df, "neutral", figures_dir / "neutral_keywords_bar.png")
    plot_evidence_bar(evidence_summary_df, "positive", figures_dir / "positive_keywords_bar.png")

    print("\nAnalysis completed.")
    print(f"Dataset name: {dataset_name}")
    print(f"Excel report saved to: {excel_file}")
    print(f"Charts saved to: {figures_dir}")

    print("\nSample results:")
    print(result_df.head())


if __name__ == "__main__":
    main()