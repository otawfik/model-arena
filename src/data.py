"""Synthetic multi-class text dataset for the arena.

Five topics (sports, tech, food, travel, finance), each with class-indicative
vocabulary mixed with generic filler words through short sentence templates.
Deterministic via numpy RandomState(seed) so tournaments are reproducible.
A small amount of cross-topic word noise keeps the task non-trivial: with
noise=0.12, 12% of samples swap one topic word for a word from another
topic, so the classes overlap and battles actually get decided.
"""
import numpy as np

CLASSES = ["sports", "tech", "food", "travel", "finance"]

TOPIC_WORDS = {
    "sports": ["goal", "match", "coach", "stadium", "victory", "penalty",
               "league", "tournament", "player", "score", "championship",
               "referee", "halftime", "trophy", "defense"],
    "tech": ["algorithm", "server", "database", "python", "neural", "cloud",
             "api", "deploy", "model", "latency", "kernel", "compiler",
             "framework", "cache", "token"],
    "food": ["recipe", "flavor", "spicy", "oven", "garlic", "sauce", "bake",
             "chef", "ingredient", "crispy", "dinner", "restaurant", "simmer",
             "seasoning", "dessert"],
    "travel": ["flight", "passport", "hotel", "beach", "itinerary", "luggage",
               "airport", "visa", "tour", "hostel", "boarding", "cruise",
               "landmark", "backpack", "resort"],
    "finance": ["stock", "dividend", "portfolio", "inflation", "bond",
                "trading", "revenue", "market", "interest", "broker", "hedge",
                "capital", "equity", "profit", "tax"],
}

FILLER = ["today", "really", "quite", "just", "very", "new", "last", "week",
          "time", "people", "think", "about", "with", "great", "best", "after",
          "morning", "story"]

TEMPLATES = [
    "{a} {b} {f} {c} {f2}",
    "{f} {a} and {b} {f2} {c}",
    "looking at {a} {f} {b} {c}",
    "{a} {f} {f2} {b} {c}",
]


def generate(n_per_class=200, seed=42, noise=0.12):
    """Return (texts, labels): n_per_class samples per topic, shuffled.

    noise: fraction of samples where one topic word is swapped for a word
    from a different topic (label stays, text gets ambiguous).
    """
    rng = np.random.RandomState(seed)
    texts, labels = [], []
    for cls in CLASSES:
        words = TOPIC_WORDS[cls]
        other = [w for c, ws in TOPIC_WORDS.items() if c != cls for w in ws]
        for _ in range(n_per_class):
            picks = list(rng.choice(words, size=3, replace=False))
            if rng.rand() < noise:
                picks[int(rng.randint(3))] = str(rng.choice(other))
            a, b, c = picks
            f, f2 = rng.choice(FILLER, size=2, replace=False)
            tmpl = TEMPLATES[int(rng.randint(len(TEMPLATES)))]
            texts.append(tmpl.format(a=a, b=b, c=c, f=f, f2=f2))
            labels.append(cls)
    order = rng.permutation(len(texts))
    return [texts[i] for i in order], [labels[i] for i in order]


def train_test_split(texts, labels, test_size=0.25, seed=7):
    """Stratified-ish split that keeps class balance via per-class shuffling."""
    rng = np.random.RandomState(seed)
    x_train, y_train, x_test, y_test = [], [], [], []
    by_class = {}
    for t, lab in zip(texts, labels):
        by_class.setdefault(lab, []).append(t)
    for lab, items in by_class.items():
        order = rng.permutation(len(items))
        items = [items[i] for i in order]
        cut = int(len(items) * (1 - test_size))
        x_train.extend(items[:cut])
        y_train.extend([lab] * cut)
        x_test.extend(items[cut:])
        y_test.extend([lab] * (len(items) - cut))
    return x_train, x_test, y_train, y_test
