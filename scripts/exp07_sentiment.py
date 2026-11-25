"""Experiment 7: IMDb Sentiment Analysis (TF-IDF+LR, LSTM, Vanilla Transformer, BERT)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def run_tfidf_lr(train_texts, y_train, test_texts, y_test):
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_tr, X_te = vec.fit_transform(train_texts), vec.transform(test_texts)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(X_tr, y_train)
    preds = clf.predict(X_te)
    acc = accuracy_score(y_test, preds)
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}
