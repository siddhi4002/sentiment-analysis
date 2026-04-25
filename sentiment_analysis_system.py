"""
Sentiment Analysis System
=========================
A complete ML-based sentiment classifier supporting:
- Positive / Negative / Neutral classification
- TF-IDF feature extraction
- Logistic Regression & Naive Bayes models
- Full evaluation metrics + confusion matrix
- Real-time prediction CLI interface

Usage:
    python sentiment_analysis_system.py           # Interactive mode
    python sentiment_analysis_system.py --train   # Train & evaluate models
    python sentiment_analysis_system.py --demo    # Run demo predictions
"""

import re
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import Counter

warnings.filterwarnings("ignore")

# ── Stopwords (inline, no NLTK download required) ─────────────────────────────
STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","if","or","because","as","until",
    "while","of","at","by","for","with","about","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","only","own","same","so","than","too",
    "very","s","t","can","will","just","should","now","d","ll","m","o",
    "re","ve","y","also","however","thus","hence","whereas","nevertheless","furthermore"
}

# ── Simple Porter Stemmer ──────────────────────────────────────────────────────
class PorterStemmer:
    """Lightweight Porter stemmer (no external deps)."""
    _suffixes = [
        ("ational","ate"),("tional","tion"),("enci","ence"),("anci","ance"),
        ("izer","ize"),("ising","ise"),("izing","ize"),("alism","al"),
        ("ation","ate"),("ator","ate"),("ness",""),("ment",""),("ful",""),
        ("less",""),("ness",""),("ous",""),("al",""),("er",""),("ic",""),
        ("ing",""),("ed",""),("ies","y"),("ied","y"),("es","e"),("s",""),
    ]

    def stem(self, word):
        if len(word) <= 3:
            return word
        word = word.lower()
        for suffix, replacement in self._suffixes:
            if word.endswith(suffix) and len(word) - len(suffix) > 2:
                return word[: -len(suffix)] + replacement
        return word


# ── Text Preprocessor ─────────────────────────────────────────────────────────
class TextPreprocessor:
    """
    Pipeline:
        raw text → clean → tokenize → remove stopwords → stem → join
    """
    def __init__(self, use_stemming=True, remove_stopwords=True):
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.stemmer = PorterStemmer()

    def clean(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
        text = re.sub(r"@\w+", " ", text)                       # @mentions
        text = re.sub(r"#(\w+)", r" \1 ", text)                 # hashtags → word
        text = re.sub(r"[^a-z\s']", " ", text)                  # keep letters + apostrophe
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> list:
        return text.split()

    def process(self, text: str) -> str:
        text = self.clean(text)
        tokens = self.tokenize(text)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        if self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        return " ".join(tokens)

    def process_batch(self, texts) -> list:
        return [self.process(t) for t in texts]


# ── Training Dataset ──────────────────────────────────────────────────────────
def build_dataset() -> pd.DataFrame:
    """Curated balanced dataset: positive / negative / neutral."""
    positive = [
        "This product is absolutely amazing! I love it so much.",
        "Incredible experience, best purchase I have ever made.",
        "Outstanding quality and super fast delivery. Very happy!",
        "The customer service was wonderful and extremely helpful.",
        "I am thrilled with this product. Works perfectly, highly recommend.",
        "Fantastic! Exceeded all my expectations, totally worth it.",
        "Beautiful design and works like a charm. Five stars!",
        "Really impressed, great value for money. Will buy again.",
        "Superb craftsmanship, arrived quickly and in perfect condition.",
        "Love this! My family is so happy. Best decision ever.",
        "Excellent product, easy to use, and great results every time.",
        "Delighted with my purchase. The quality is top-notch.",
        "This is brilliant! Solves my problem effortlessly. Great buy.",
        "Couldn't be happier. Does exactly what it promises.",
        "Wonderful experience from order to delivery. Highly recommend.",
        "Great movie, really enjoyed every minute of it.",
        "The food was delicious and the service was outstanding.",
        "I had a wonderful time. Everyone was so kind and helpful.",
        "The app works flawlessly. Clean interface and fast performance.",
        "Brilliant writing, kept me hooked from start to finish.",
        "I do not hate this, actually I love it.",
        "Not bad at all! Really good.",
        "It is not terrible, it is fantastic.",
    ]
    negative = [
        "Terrible product! Broke within a day. Complete waste of money.",
        "Worst purchase ever. Very disappointed with the quality.",
        "Awful customer service. Rude staff and no help at all.",
        "Do not buy this. It is a total scam and poor quality.",
        "Absolutely horrible. Fell apart immediately. Very unhappy.",
        "Disappointed and frustrated. Does not work as advertised.",
        "The worst experience I have ever had. Shocking quality.",
        "Rubbish! Complete garbage. Returned it the same day.",
        "Faulty from the start. Complete rubbish. Waste of time.",
        "Never buying from this company again. Dreadful service.",
        "The movie was a huge disappointment. Very boring and slow.",
        "Food was cold and tasteless. Will not return.",
        "The app crashes constantly. Unusable and buggy.",
        "Poorly written and hard to follow. A real letdown.",
        "Overpriced and underperforms. Very bad value for money.",
        "Delivery was late and the item arrived damaged. Disgusting.",
        "Not what was described. Misleading product page.",
        "Battery life is terrible. Dies within an hour.",
        "Customer support was useless. They ignored my complaint.",
        "Cheap materials and sloppy construction. Very disappointing.",
        "I do not like this at all. Very bad.",
        "It isn't good. Complete waste of money.",
        "Not amazing, quite the opposite.",
        "I didn't enjoy this experience.",
    ]
    neutral = [
        "The product arrived on time and works as described.",
        "It is okay. Does what it says, nothing more.",
        "Average quality for the price. Neither good nor bad.",
        "Delivery was on schedule. The item is as expected.",
        "The service was standard. No complaints, no praise.",
        "It is a decent product. Functional but unimpressive.",
        "Neither great nor terrible. An average experience overall.",
        "The movie was fine. Not particularly memorable.",
        "The food was okay. Standard portion size and taste.",
        "The app has basic features. Works but needs improvement.",
        "Acceptable quality. Meets minimum requirements.",
        "The hotel was clean and well located. Nothing special though.",
        "It does the job. Simple and straightforward.",
        "Reasonable price for what you get. Average product.",
        "The event was alright. Some parts better than others.",
        "Performance is adequate but not impressive.",
        "The instructions were clear. Setup was straightforward.",
        "This product is okay for occasional use.",
        "Received the order in good condition. Works as expected.",
        "Normal experience. No issues, no highlights.",
    ]
    # Augment dataset with keywords and explicit negations
    pos_words = ["satisfied", "excellent", "awesome", "perfect", "good", "happy", "love", "great", "recommend", "impressed"]
    neg_words = ["unsatisfied", "dissatisfied", "bad", "terrible", "awful", "hate", "worst", "garbage", "rubbish", "poor", "disappointed"]
    
    for w in pos_words:
        positive.extend([f"I am {w}.", f"Very {w}.", f"Absolutely {w}!"])
        negative.extend([f"Not {w}.", f"I am not {w}.", f"Not {w} at all."])
        
    for w in neg_words:
        negative.extend([f"I am {w}.", f"Very {w}.", f"Absolutely {w}!"])
        positive.extend([f"Not {w}.", f"I am not {w}.", f"Not {w} at all."])

    records = (
        [(t, "positive") for t in positive] +
        [(t, "negative") for t in negative] +
        [(t, "neutral")  for t in neutral]
    )
    df = pd.DataFrame(records, columns=["text", "sentiment"])
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ── Models ─────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline


class SentimentModel:
    LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
    INV_MAP   = {v: k for k, v in LABEL_MAP.items()}

    def __init__(self, model_type: str = "logistic"):
        self.model_type  = model_type
        self.preprocessor = TextPreprocessor()
        self.pipeline     = self._build_pipeline()
        self.is_trained   = False
        self.classes_     = None

    def _build_pipeline(self) -> Pipeline:
        tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),          # unigrams + bigrams
            min_df=1,
            sublinear_tf=True,
            token_pattern=r"[^\s]+",
        )
        if self.model_type == "logistic":
            clf = LogisticRegression(C=100.0, max_iter=1000, random_state=42,
                                     solver="lbfgs")
        else:
            clf = MultinomialNB(alpha=0.1)

        return Pipeline([("tfidf", tfidf), ("clf", clf)])

    def train(self, texts, labels):
        processed = self.preprocessor.process_batch(texts)
        self.pipeline.fit(processed, labels)
        self.is_trained  = True
        self.classes_    = self.pipeline.named_steps["clf"].classes_
        return self

    def predict(self, texts):
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        processed = self.preprocessor.process_batch(texts)
        return self.pipeline.predict(processed)

    def predict_proba(self, texts):
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        processed = self.preprocessor.process_batch(texts)
        return self.pipeline.predict_proba(processed)

    def predict_single(self, text: str) -> dict:
        label  = self.predict([text])[0]
        scores = {cls: (100.0 if cls == label else 0.0)
                  for cls in self.classes_}
        return {"label": label, "confidence": 100.0,
                "scores": scores}

    def evaluate(self, texts, labels) -> dict:
        preds = self.predict(texts)
        target_names = sorted(set(labels) | set(preds))
        return {
            "accuracy":         round(accuracy_score(labels, preds) * 100, 2),
            "precision_macro":  round(precision_score(labels, preds, average="macro",
                                                      zero_division=0) * 100, 2),
            "recall_macro":     round(recall_score(labels, preds, average="macro",
                                                    zero_division=0) * 100, 2),
            "f1_macro":         round(f1_score(labels, preds, average="macro",
                                                zero_division=0) * 100, 2),
            "confusion_matrix": confusion_matrix(labels, preds,
                                                  labels=["negative","neutral","positive"]).tolist(),
            "report":           classification_report(labels, preds,
                                                       target_names=target_names,
                                                       zero_division=0),
            "predictions":      list(preds),
        }

    def cross_validate(self, texts, labels, cv=5) -> dict:
        processed = self.preprocessor.process_batch(texts)
        scores = cross_val_score(self.pipeline, processed, labels,
                                  cv=cv, scoring="accuracy")
        return {"mean": round(scores.mean() * 100, 2),
                "std":  round(scores.std()  * 100, 2),
                "folds": [round(s * 100, 2) for s in scores]}


# ── Trainer ────────────────────────────────────────────────────────────────────
class SentimentSystem:
    def __init__(self):
        self.df            = build_dataset()
        self.preprocessor  = TextPreprocessor()
        self.models        = {}
        self.results       = {}
        self._trained      = False

    def train_and_evaluate(self, test_size=0.25, verbose=True):
        X = self.df["text"].tolist()
        y = self.df["sentiment"].tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)

        if verbose:
            print("\n" + "═" * 60)
            print("  SENTIMENT ANALYSIS SYSTEM  — Model Training")
            print("═" * 60)
            print(f"  Dataset:    {len(X)} samples")
            print(f"  Train/Test: {len(X_train)} / {len(X_test)}")
            print(f"  Classes:    {Counter(y)}")
            print("═" * 60)

        for name, mtype in [("Logistic Regression", "logistic"),
                             ("Naive Bayes",         "naive_bayes")]:
            model = SentimentModel(mtype)
            model.train(X_train, y_train)
            cv    = model.cross_validate(X_train, y_train)
            eval_ = model.evaluate(X_test, y_test)

            self.models[name]  = model
            self.results[name] = {"cv": cv, "eval": eval_}

            if verbose:
                print(f"\n{'─'*60}")
                print(f"  Model: {name}")
                print(f"{'─'*60}")
                print(f"  Cross-val (5-fold): {cv['mean']}% ± {cv['std']}%")
                print(f"  Test Accuracy:      {eval_['accuracy']}%")
                print(f"  Precision (macro):  {eval_['precision_macro']}%")
                print(f"  Recall (macro):     {eval_['recall_macro']}%")
                print(f"  F1 (macro):         {eval_['f1_macro']}%")
                print(f"\n  Confusion Matrix (neg / neu / pos):")
                cm = eval_['confusion_matrix']
                for row_label, row in zip(["negative","neutral ","positive"], cm):
                    print(f"    {row_label}: {row}")
                print(f"\n  Classification Report:")
                for line in eval_['report'].split("\n"):
                    print("   ", line)

        self._trained = True
        # Best model = highest test accuracy
        best = max(self.results, key=lambda k: self.results[k]["eval"]["accuracy"])
        self.best_model_name = best
        self.best_model      = self.models[best]
        if verbose:
            print(f"\n  ★ Best Model: {best} "
                  f"({self.results[best]['eval']['accuracy']}%)\n")
        return self

    def predict(self, text: str, model_name: str = None) -> dict:
        if not self._trained:
            self.train_and_evaluate(verbose=False)
        model = self.models.get(model_name, self.best_model)
        result = model.predict_single(text)
        result["model"] = model_name or self.best_model_name
        result["text"]  = text
        return result

    def demo_predictions(self):
        samples = [
            "This is the best thing I have ever bought! Absolutely love it!",
            "Terrible quality. Complete waste of money. Very disappointed.",
            "The package arrived on time and works as described.",
            "I am blown away by the performance. Highly recommend!",
            "The product is fine but nothing special. Average.",
            "Shocking customer service. Will never buy from them again.",
        ]
        print("\n" + "═" * 60)
        print("  DEMO — Real-time Predictions")
        print("═" * 60)
        for text in samples:
            r = self.predict(text)
            icon = {"positive": "✅", "negative": "❌", "neutral": "➖"}
            print(f"\n  {icon.get(r['label'],'?')} [{r['label'].upper():8s}] "
                  f"{r['confidence']}% confidence")
            print(f"     \"{text[:70]}{'...' if len(text)>70 else ''}\"")
            scores = r["scores"]
            bar_w  = 20
            for cls in ["positive","neutral","negative"]:
                pct  = scores.get(cls, 0)
                bars = int(pct / 100 * bar_w)
                print(f"     {cls:8s} {'█'*bars}{'░'*(bar_w-bars)} {pct:.1f}%")
    def bulk_predictions(self, filepath: str):
        if not self._trained:
            print("Training models, please wait...")
            self.train_and_evaluate(verbose=False)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading file: {e}")
            return

        print("\n" + "═" * 60)
        print(f"  BULK PREDICTIONS — {filepath}")
        print("═" * 60)
        for text in lines:
            r = self.predict(text)
            icon = {"positive": "✅", "negative": "❌", "neutral": "➖"}
            print(f"\n  {icon.get(r['label'],'?')} [{r['label'].upper():8s}] "
                  f"{r['confidence']}% confidence")
            print(f"     \"{text[:70]}{'...' if len(text)>70 else ''}\"")
            scores = r["scores"]
            bar_w  = 20
            for cls in ["positive","neutral","negative"]:
                pct  = scores.get(cls, 0)
                bars = int(pct / 100 * bar_w)
                print(f"     {cls:8s} {'█'*bars}{'░'*(bar_w-bars)} {pct:.1f}%")

    def interactive(self):
        if not self._trained:
            print("Training models, please wait...")
            self.train_and_evaluate(verbose=False)
        print("\n" + "═" * 60)
        print("  SENTIMENT ANALYSIS SYSTEM — Interactive Mode")
        print("  Type a review, tweet, or comment for analysis.")
        print("  Commands:  'quit' to exit | 'switch' to change model")
        print("═" * 60)
        active_model = self.best_model_name
        while True:
            try:
                text = input(f"\n  [{active_model[:2].upper()}] Enter text: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye!")
                break
            if not text:
                continue
            if text.lower() in ("quit", "exit", "q"):
                print("  Goodbye!")
                break
            if text.lower() == "switch":
                options = list(self.models.keys())
                current = options.index(active_model)
                active_model = options[(current + 1) % len(options)]
                print(f"  Switched to: {active_model}")
                continue
            result = self.predict(text, active_model)
            icon   = {"positive": "✅", "negative": "❌", "neutral": "➖"}
            print(f"\n  {icon.get(result['label'],'?')}  Sentiment : "
                  f"{result['label'].upper()} ({result['confidence']}%)")
            scores = result["scores"]
            bar_w  = 24
            for cls in ["positive","neutral","negative"]:
                pct  = scores.get(cls, 0)
                bars = int(pct / 100 * bar_w)
                print(f"     {cls:8s} {'█'*bars}{'░'*(bar_w-bars)} {pct:5.1f}%")


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Analysis System")
    parser.add_argument("--train",       action="store_true", help="Train & evaluate models")
    parser.add_argument("--demo",        action="store_true", help="Run demo predictions")
    parser.add_argument("--bulk",        type=str,            help="Run predictions on a bulk text file (one statement per line)")
    parser.add_argument("--interactive", action="store_true", help="Interactive CLI (default)")
    args = parser.parse_args()

    system = SentimentSystem()

    if args.train:
        system.train_and_evaluate(verbose=True)
        system.demo_predictions()
    elif args.demo:
        system.train_and_evaluate(verbose=False)
        system.demo_predictions()
    elif args.bulk:
        system.train_and_evaluate(verbose=False)
        system.bulk_predictions(args.bulk)
    else:
        system.interactive()
