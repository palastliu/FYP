import re
from typing import Dict, List, Set


# Aspect configuration
# You can expand or edit these keywords at any time.
ASPECT_CONFIG: Dict[str, Set[str]] = {
    "service": {
        "service", "support", "staff", "reply", "response", "customer",
        "team", "help", "refund", "contact", "assistance", "seller",
        "store", "shop", "agent"
    },

    "performance": {
        "crash", "freeze", "bug", "error", "slow", "lag", "battery",
        "drain", "loading", "performance", "speed", "stability", "memory",
        "glitch", "fault", "issue"
    },

    "usability": {
        "easy", "use", "using", "simple", "difficult", "hard", "user",
        "friendly", "convenient", "install", "setup", "navigation",
        "operate", "access", "understand"
    },

    "interface": {
        "design", "interface", "layout", "screen", "ui", "look",
        "theme", "color", "display", "home", "menu", "visual",
        "appearance"
    },

    "pricing": {
        "price", "pricing", "payment", "subscription", "premium",
        "free", "version", "pro", "cost", "paid", "purchase", "buy",
        "value", "money", "cheap", "expensive"
    },

    "feature": {
        "calendar", "task", "widget", "feature", "option", "list",
        "tracker", "reminder", "notification", "sync", "integration",
        "tool", "function", "setting", "filter"
    },

    "quality": {
        "quality", "material", "fabric", "durable", "durability",
        "cheaply", "stitching", "construction", "zipper", "button",
        "thread", "seam"
    },

    "fit_size": {
        "size", "fit", "small", "large", "tight", "loose", "short",
        "long", "length", "waist", "shoulder", "sleeve", "narrow",
        "wide"
    },

    "style_appearance": {
        "style", "stylish", "beautiful", "ugly", "cute", "dress",
        "shirt", "skirt", "jeans", "jacket", "color", "pattern",
        "shape", "look"
    },

    "comfort": {
        "comfortable", "comfort", "soft", "itchy", "scratchy",
        "breathable", "warm", "hot", "heavy", "lightweight"
    },

    "delivery": {
        "delivery", "shipping", "ship", "arrive", "arrived", "late",
        "delay", "package", "packaging", "return", "exchange"
    },

    "reliability": {
        "reliable", "reliability", "consistent", "stable", "broken",
        "break", "fail", "failure", "working", "works"
    },

    "account_billing": {
        "account", "login", "password", "billing", "invoice",
        "charge", "charged", "cancel", "renewal"
    },

    "security_privacy": {
        "privacy", "security", "secure", "permission", "permissions",
        "tracking", "data", "personal", "unsafe"
    },

    "content": {
        "content", "article", "video", "image", "music", "course",
        "lesson", "tutorial", "document"
    },
}


OVERALL_TERMS = {
    "app", "application", "software", "tool", "program",
    "product", "item", "dress", "shirt", "skirt", "jeans",
    "jacket", "top", "purchase", "order"
}

POSITIVE_CUES = {
    "good", "great", "excellent", "amazing", "awesome", "easy", "helpful",
    "smooth", "love", "like", "best", "nice", "useful", "fast", "stable",
    "recommend", "perfect", "happy", "well", "better", "clean", "simple",
    "beautiful", "comfortable"
}

NEGATIVE_CUES = {
    "bad", "poor", "slow", "terrible", "awful", "disappointing", "worst",
    "crash", "freeze", "drain", "bug", "issue", "problem", "broken",
    "late", "hard", "difficult", "annoying", "waste", "frustrating",
    "unhelpful", "fail", "error", "wrong", "expensive", "itchy",
    "small", "large", "tight", "loose"
}


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def simple_tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]+", normalize_text(text))


def map_aspect(evidence_phrase: str) -> str:
    """
    Generic, extensible aspect mapping.

    Priority:
    1. Match configurable aspect keywords
    2. Map generic praise/complaint to overall_experience
    3. Fallback to general
    """
    phrase = normalize_text(evidence_phrase)
    words = set(simple_tokens(phrase))

    best_aspect = None
    best_score = 0

    for aspect, keywords in ASPECT_CONFIG.items():
        score = len(words & keywords)
        if score > best_score:
            best_score = score
            best_aspect = aspect

    if best_aspect is not None and best_score > 0:
        return best_aspect

    if (words & OVERALL_TERMS) and ((words & POSITIVE_CUES) or (words & NEGATIVE_CUES)):
        return "overall_experience"

    return "general"


def aspect_priority(aspect: str) -> int:
    """
    Lower number = higher display priority.
    """
    if aspect in ASPECT_CONFIG:
        return 0
    if aspect == "overall_experience":
        return 1
    return 2