"""Model registry: six classifiers wrapped in TF-IDF pipelines.

Every contender is a scikit-learn Pipeline(TfidfVectorizer, classifier) so
training and prediction share one interface for the arena.
"""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DISPLAY_NAMES = {
    "naive-bayes": "Naive Bayes",
    "logreg": "Logistic Regression",
    "linear-svm": "Linear SVM",
    "random-forest": "Random Forest",
    "knn": "k-Nearest Neighbors",
    "dummy": "Dummy (majority class)",
}


def _pipe(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2000)),
        ("clf", clf),
    ])


MODEL_BUILDERS = {
    "naive-bayes": lambda: _pipe(MultinomialNB()),
    "logreg": lambda: _pipe(LogisticRegression(max_iter=2000)),
    "linear-svm": lambda: _pipe(LinearSVC()),
    "random-forest": lambda: _pipe(RandomForestClassifier(
        n_estimators=100, random_state=7)),
    "knn": lambda: _pipe(KNeighborsClassifier(n_neighbors=5)),
    "dummy": lambda: _pipe(DummyClassifier(strategy="most_frequent")),
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
