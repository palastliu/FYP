from src.fuzzy_system import infer_sentiment

examples = [
    (0.10, 0.20, 0.70),
    (0.65, 0.25, 0.10),
    (0.30, 0.50, 0.20),
]

for i, (p_neg, p_neu, p_pos) in enumerate(examples, start=1):
    score, label, fired_rules = infer_sentiment(p_neg, p_neu, p_pos)
    print(f"Example {i}: score={score:.4f}, label={label}")
    print("Top fired rules:")
    for rule in fired_rules[:3]:
        print(rule)
    print("-" * 50)