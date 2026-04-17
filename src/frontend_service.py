from pathlib import Path
import re

import pandas as pd
import torch
import spacy
from scipy.special import softmax
from transformers import BertTokenizer, BertForSequenceClassification

from src.config import BERT_MODEL_DIR, MAX_LEN, OUTPUT_DIR
from src.fuzzy_system import infer_sentiment, get_default_params
from src.aspect_mapper import map_aspect
from src.app_batch_analyzer import (
    classify_evidence_phrases,
    generate_risk_level,
    generate_action_hint,
    create_summary_sheet,
    build_evidence_summary,
    save_excel_report,
    plot_label_distribution_pie,
    plot_evidence_bar,
)

try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    NLP = None


def make_safe_name(name: str) -> str:
    name = Path(str(name)).stem.lower().strip()
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "dataset"


def build_frontend_output_paths(uploaded_name: str):
    dataset_name = make_safe_name(uploaded_name)
    report_dir = OUTPUT_DIR / "frontend_outputs" / f"{dataset_name}_output"
    figures_dir = report_dir / "figures"
    excel_file = report_dir / "comment_analysis_report.xlsx"
    return dataset_name, report_dir, figures_dir, excel_file


def load_model():
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict_probs(text, model, tokenizer, device):
    encoded = tokenizer(
        str(text).strip(),
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


def build_single_explanation(final_label, aspect, p_neg, p_neu, p_pos, evidence_rows):
    probs = {
        "negative": float(p_neg),
        "neutral": float(p_neu),
        "positive": float(p_pos),
    }
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_label, top_prob = sorted_probs[0]
    second_label, second_prob = sorted_probs[1]
    margin = top_prob - second_prob

    if evidence_rows:
        top_evidence = evidence_rows[0]["evidence_phrase"]
        evidence_text = f'The strongest supporting evidence phrase is "{top_evidence}".'
    else:
        evidence_text = "No strong evidence phrase was extracted from this short comment."

    aspect_text = (
        f'The comment is mainly interpreted under the "{aspect}" aspect.'
        if aspect != "general"
        else "No highly specific aspect was detected, so the comment is assigned to a general aspect."
    )

    if final_label == "positive":
        tone_text = (
            f'The comment is classified as positive because the positive probability ({p_pos:.3f}) '
            f'is higher than the neutral ({p_neu:.3f}) and negative ({p_neg:.3f}) probabilities.'
        )
    elif final_label == "negative":
        tone_text = (
            f'The comment is classified as negative because the negative probability ({p_neg:.3f}) '
            f'is higher than the neutral ({p_neu:.3f}) and positive ({p_pos:.3f}) probabilities.'
        )
    else:
        tone_text = (
            f'The comment is classified as neutral because the neutral probability ({p_neu:.3f}) '
            f'is the strongest or the overall sentiment is relatively balanced.'
        )

    confidence_text = (
        f"The decision margin over the next-closest class is {margin:.3f}, "
        f"which suggests a {'clearer' if margin >= 0.20 else 'milder'} preference for this label."
    )

    return f"{tone_text} {aspect_text} {evidence_text} {confidence_text}"


def analyze_single_comment(text, model, tokenizer, device):
    params = get_default_params()

    p_neg, p_neu, p_pos = predict_probs(text, model, tokenizer, device)

    final_score, final_label, fired_rules = infer_sentiment(
        p_neg=p_neg,
        p_neu=p_neu,
        p_pos=p_pos,
        params=params
    )

    top_rule = "fallback_rule"
    if fired_rules:
        top_rule = max(fired_rules, key=lambda x: x["alpha"])["rule"]

    evidence_rows = classify_evidence_phrases(text)
    matched_evidence = [e for e in evidence_rows if e["evidence_label"] == final_label]

    if matched_evidence:
        main_aspect = map_aspect(matched_evidence[0]["evidence_phrase"])
    else:
        main_aspect = "general"

    explanation = build_single_explanation(
        final_label=final_label,
        aspect=main_aspect,
        p_neg=p_neg,
        p_neu=p_neu,
        p_pos=p_pos,
        evidence_rows=matched_evidence
    )

    return {
        "comment": text,
        "p_neg": p_neg,
        "p_neu": p_neu,
        "p_pos": p_pos,
        "final_score": final_score,
        "final_label": final_label,
        "aspect": main_aspect,
        "risk_level": generate_risk_level(final_label),
        "action_hint": generate_action_hint(final_label),
        "explanation": explanation,
        "top_rule": top_rule,
        "evidence_rows": matched_evidence
    }


def run_batch_analysis(df, uploaded_name, model, tokenizer, device):
    params = get_default_params()

    if "comment" not in df.columns:
        raise ValueError("Input CSV must contain a 'comment' column.")

    if "review_id" not in df.columns:
        df = df.copy()
        df.insert(0, "review_id", range(1, len(df) + 1))

    dataset_name, output_dir, figures_dir, excel_file = build_frontend_output_paths(uploaded_name)

    results = []
    for _, row in df.iterrows():
        text = str(row["comment"]).strip()

        p_neg, p_neu, p_pos = predict_probs(text, model, tokenizer, device)
        final_score, final_label, fired_rules = infer_sentiment(
            p_neg=p_neg,
            p_neu=p_neu,
            p_pos=p_pos,
            params=params
        )

        top_rule = "fallback_rule"
        if fired_rules:
            top_rule = max(fired_rules, key=lambda x: x["alpha"])["rule"]

        evidence_rows = classify_evidence_phrases(text)
        matched = [e for e in evidence_rows if e["evidence_label"] == final_label]
        aspect = map_aspect(matched[0]["evidence_phrase"]) if matched else "general"

        results.append({
            "review_id": row["review_id"],
            "comment": text,
            "final_label": final_label,
            "aspect": aspect,
            "risk_level": generate_risk_level(final_label),
            "action_hint": generate_action_hint(final_label),
            "top_rule": top_rule,
        })

    result_df = pd.DataFrame(results)

    evidence_summary_df = build_evidence_summary(result_df, top_n=10)
    summary_df = create_summary_sheet(result_df, original_df=df)

    output_dir.mkdir(parents=True, exist_ok=True)
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

    return result_df, summary_df, evidence_summary_df, excel_file, figures_dir, dataset_name, output_dir