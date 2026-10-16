"""Model registry: six classifiers wrapped in TF-IDF pipelines.

Every contender is a scikit-learn Pipeline(TfidfVectorizer, classifier) so
training and prediction share one interface for the arena. The vectorizer and
hyperparameters are tuned for long, noisy forum posts (20 Newsgroups):
sublinear term frequency, unigrams + bigrams, and document-frequency cut-offs.
"""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DISPLAY_NAMES = {
    "naive-bayes": "Complement Naive Bayes",
    "logreg": "Logistic Regression",
    "linear-svm": "Linear SVM",
    "random-forest": "Random Forest",
    "knn": "k-Nearest Neighbors",
    "dummy": "Dummy (majority class)",
}


def _pipe(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=3,
                                  max_df=0.5, stop_words="english", max_features=40000,
                                  dtype=np.float32)),
        ("clf", clf),
    ])


MODEL_BUILDERS = {
    "naive-bayes": lambda: _pipe(ComplementNB(alpha=0.3)),
    "logreg": lambda: _pipe(LogisticRegression(C=10, max_iter=3000)),
    "linear-svm": lambda: _pipe(LinearSVC(C=0.5)),
    "random-forest": lambda: _pipe(RandomForestClassifier(
        n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=7)),
    "knn": lambda: _pipe(KNeighborsClassifier(n_neighbors=10, metric="cosine", weights="distance")),
    "dummy": lambda: _pipe(DummyClassifier(strategy="prior")),
}


def names():
    """Registered model keys."""
    return list(MODEL_BUILDERS)


def build(name):
    """Fresh unfitted pipeline for a registered model key."""
    if name not in MODEL_BUILDERS:
        raise KeyError(f"unknown model: {name!r}")
    return MODEL_BUILDERS[name]()


def train(name, x_train, y_train):
    """Build and fit a model on the given training data."""
    model = build(name)
    model.fit(x_train, y_train)
    return model


def predict(model, x):
    """Predict class labels for a list of texts."""
    return list(model.predict(x))


def confidence(model, x):
    """Top-class probability per text, or None for models without probabilities.

    Linear SVM only exposes margins, which aren't probabilities, so it reports
    None rather than a number that looks comparable but isn't.
    """
    if not hasattr(model, "predict_proba"):
        return [None] * len(x)
    try:
        return [float(p) for p in model.predict_proba(x).max(axis=1)]
    except AttributeError:
        return [None] * len(x)
