from copy import deepcopy


DEFAULT_PARAMS = {
    "neg": {
        "low": {"a": 0.30, "b": 0.48},
        "medium": {"a": 0.20, "b": 0.48, "c": 0.78},
        "high": {"a": 0.48, "b": 0.70},
    },
    "neu": {
        "low": {"a": 0.30, "b": 0.48},
        "medium": {"a": 0.20, "b": 0.48, "c": 0.78},
        "high": {"a": 0.48, "b": 0.70},
    },
    "pos": {
        "low": {"a": 0.30, "b": 0.48},
        "medium": {"a": 0.20, "b": 0.48, "c": 0.78},
        "high": {"a": 0.48, "b": 0.70},
    },
    "output": {
        "negative": {"z_min": 0.00, "z_max": 0.35},
        "neutral_low": {"z_min": 0.35, "z_max": 0.50},
        "neutral_high": {"z_min": 0.50, "z_max": 0.65},
        "positive": {"z_min": 0.65, "z_max": 1.00},
    }
}


RULE_BASE = [
    ("low", "low", "low", "neutral_low"),
    ("low", "low", "medium", "positive"),
    ("low", "low", "high", "positive"),

    ("low", "medium", "low", "neutral_low"),
    ("low", "medium", "medium", "neutral_high"),
    ("low", "medium", "high", "positive"),

    ("low", "high", "low", "neutral_low"),
    ("low", "high", "medium", "neutral_high"),
    ("low", "high", "high", "positive"),

    ("medium", "low", "low", "negative"),
    ("medium", "low", "medium", "neutral_low"),
    ("medium", "low", "high", "positive"),

    ("medium", "medium", "low", "negative"),
    ("medium", "medium", "medium", "neutral_low"),
    ("medium", "medium", "high", "neutral_high"),

    ("medium", "high", "low", "neutral_low"),
    ("medium", "high", "medium", "neutral_high"),
    ("medium", "high", "high", "neutral_high"),

    ("high", "low", "low", "negative"),
    ("high", "low", "medium", "negative"),
    ("high", "low", "high", "neutral_high"),

    ("high", "medium", "low", "negative"),
    ("high", "medium", "medium", "negative"),
    ("high", "medium", "high", "neutral_high"),

    ("high", "high", "low", "negative"),
    ("high", "high", "medium", "neutral_low"),
    ("high", "high", "high", "neutral_high"),
]


def get_default_params():
    return deepcopy(DEFAULT_PARAMS)


def low_membership(x, a, b):
    if x <= a:
        return 1.0
    if a < x < b:
        return (b - x) / (b - a)
    return 0.0


def medium_membership(x, a, b, c):
    if a <= x <= b:
        return (x - a) / (b - a)
    if b < x <= c:
        return (c - x) / (c - b)
    return 0.0


def high_membership(x, a, b):
    if x <= a:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)
    return 1.0


def fuzzify_input(x, variable_params):
    x = float(x)
    return {
        "low": low_membership(x, variable_params["low"]["a"], variable_params["low"]["b"]),
        "medium": medium_membership(
            x,
            variable_params["medium"]["a"],
            variable_params["medium"]["b"],
            variable_params["medium"]["c"],
        ),
        "high": high_membership(x, variable_params["high"]["a"], variable_params["high"]["b"]),
    }


def tsukamoto_z_negative(alpha, output_params):
    z_min = output_params["negative"]["z_min"]
    z_max = output_params["negative"]["z_max"]
    return z_max - (z_max - z_min) * alpha


def tsukamoto_z_neutral_low(alpha, output_params):
    z_min = output_params["neutral_low"]["z_min"]
    z_max = output_params["neutral_low"]["z_max"]
    return z_min + (z_max - z_min) * alpha


def tsukamoto_z_neutral_high(alpha, output_params):
    z_min = output_params["neutral_high"]["z_min"]
    z_max = output_params["neutral_high"]["z_max"]
    return z_max - (z_max - z_min) * alpha


def tsukamoto_z_positive(alpha, output_params):
    z_min = output_params["positive"]["z_min"]
    z_max = output_params["positive"]["z_max"]
    return z_min + (z_max - z_min) * alpha


def build_rule_base():
    return RULE_BASE


def fallback_prediction(p_neg, p_neu, p_pos):
    probs = {
        "negative": float(p_neg),
        "neutral": float(p_neu),
        "positive": float(p_pos),
    }
    label = max(probs, key=probs.get)

    if label == "negative":
        score = 0.20
    elif label == "neutral":
        score = 0.50
    else:
        score = 0.80

    return score, label


def infer_sentiment(p_neg, p_neu, p_pos, params=None):
    if params is None:
        params = get_default_params()

    neg_fuzzy = fuzzify_input(p_neg, params["neg"])
    neu_fuzzy = fuzzify_input(p_neu, params["neu"])
    pos_fuzzy = fuzzify_input(p_pos, params["pos"])

    rules = build_rule_base()

    weighted_sum = 0.0
    alpha_sum = 0.0
    fired_rules = []

    for neg_level, neu_level, pos_level, output_label in rules:
        alpha = min(
            neg_fuzzy[neg_level],
            neu_fuzzy[neu_level],
            pos_fuzzy[pos_level]
        )

        if alpha > 0:
            if output_label == "negative":
                z = tsukamoto_z_negative(alpha, params["output"])
            elif output_label == "neutral_low":
                z = tsukamoto_z_neutral_low(alpha, params["output"])
            elif output_label == "neutral_high":
                z = tsukamoto_z_neutral_high(alpha, params["output"])
            else:
                z = tsukamoto_z_positive(alpha, params["output"])

            weighted_sum += alpha * z
            alpha_sum += alpha

            fired_rules.append({
                "rule": f"IF neg is {neg_level} AND neu is {neu_level} AND pos is {pos_level} THEN {output_label}",
                "alpha": alpha,
                "z": z,
                "output": output_label
            })

    if alpha_sum == 0:
        final_score, label = fallback_prediction(p_neg, p_neu, p_pos)
        return final_score, label, fired_rules

    final_score = weighted_sum / alpha_sum

    if final_score < params["output"]["negative"]["z_max"]:
        label = "negative"
    elif final_score < params["output"]["neutral_high"]["z_max"]:
        label = "neutral"
    else:
        label = "positive"

    return final_score, label, fired_rules