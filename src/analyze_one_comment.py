import torch
from scipy.special import softmax
from transformers import BertTokenizer, BertForSequenceClassification

from src.config import BERT_MODEL_DIR, MAX_LEN
from src.fuzzy_system import infer_sentiment, get_default_params


def load_model_and_tokenizer(model_dir):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, device


def preprocess_text(text: str) -> str:
    text = text.strip()
    return text


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


def generate_explanation(p_neg, p_neu, p_pos, final_label, top_rule):
    if final_label == "positive":
        return (
            f"The comment is classified as positive because the positive score "
            f"({p_pos:.3f}) is stronger than the negative ({p_neg:.3f}) and "
            f"neutral ({p_neu:.3f}) scores. The activated fuzzy rule also supports a positive decision."
        )
    elif final_label == "negative":
        return (
            f"The comment is classified as negative because the negative score "
            f"({p_neg:.3f}) is stronger than the positive ({p_pos:.3f}) and "
            f"neutral ({p_neu:.3f}) scores. The activated fuzzy rule also supports a negative decision."
        )
    else:
        return (
            f"The comment is classified as neutral because the sentiment scores are relatively balanced "
            f"(neg={p_neg:.3f}, neu={p_neu:.3f}, pos={p_pos:.3f}), which leads the Tsukamoto fuzzy system "
            f"to produce a neutral decision."
        )


def generate_risk_level(final_label):
    if final_label == "negative":
        return "High"
    elif final_label == "neutral":
        return "Medium"
    return "Low"


def generate_action_hint(final_label):
    if final_label == "negative":
        return "Immediate review recommended. Check for service or product issues behind this complaint."
    elif final_label == "neutral":
        return "Monitor this feedback. It may indicate a potential improvement area."
    return "Positive feedback. Can be used as supporting evidence of customer satisfaction."


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

    explanation = generate_explanation(
        p_neg=p_neg,
        p_neu=p_neu,
        p_pos=p_pos,
        final_label=final_label,
        top_rule=top_rule
    )

    risk_level = generate_risk_level(final_label)
    action_hint = generate_action_hint(final_label)

    return {
        "text": clean_text,
        "p_neg": p_neg,
        "p_neu": p_neu,
        "p_pos": p_pos,
        "final_score": final_score,
        "final_label": final_label,
        "top_rule": top_rule,
        "explanation": explanation,
        "risk_level": risk_level,
        "action_hint": action_hint,
    }


def main():
    print("Loading trained BERT model...")
    model, tokenizer, device = load_model_and_tokenizer(BERT_MODEL_DIR)
    print(f"Using device: {device}")

    params = get_default_params()

    print("\nEnter a comment to analyze.")
    user_text = input("Comment: ").strip()

    if not user_text:
        print("No input provided.")
        return

    result = analyze_comment(
        text=user_text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        params=params
    )

    print("\n===== Analysis Result =====")
    print(f"Comment      : {result['text']}")
    print(f"p_neg        : {result['p_neg']:.4f}")
    print(f"p_neu        : {result['p_neu']:.4f}")
    print(f"p_pos        : {result['p_pos']:.4f}")
    print(f"Final score  : {result['final_score']:.4f}")
    print(f"Final label  : {result['final_label']}")
    print(f"Top rule     : {result['top_rule']}")
    print(f"Risk level   : {result['risk_level']}")
    print(f"Action hint  : {result['action_hint']}")
    print(f"\nExplanation  : {result['explanation']}")


if __name__ == "__main__":
    main()