"""
Train the ASL letter classifier on data recorded by
scripts/collect_data.py.

Pipeline (see docs/learning/06_asl_classifier.md):
    data/asl_landmarks.csv
     -> Features (63 columns) + Labels
     -> Train/test split
     -> Random Forest
     -> Evaluation (REAL, measured metrics -- never fabricated)
     -> models/asl_classifier.pkl

Run:
    venv/Scripts/python.exe scripts/train_asl.py
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from config import ASL_LANDMARKS_CSV_PATH, ASL_MODEL_PATH  # noqa: E402


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    with open(ASL_LANDMARKS_CSV_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header row
        rows = list(reader)
    features = np.array([[float(value) for value in row[:-1]] for row in rows])
    labels = np.array([row[-1] for row in rows])
    return features, labels


def main() -> None:
    if not ASL_LANDMARKS_CSV_PATH.exists():
        print(f"No dataset found at {ASL_LANDMARKS_CSV_PATH}.")
        print("Run scripts/collect_data.py first to record your own ASL letters.")
        return

    X, y = load_dataset()
    unique_labels = sorted(set(y))
    print(f"Loaded {len(X)} samples across {len(unique_labels)} letters: {' '.join(unique_labels)}")

    if len(unique_labels) < 2:
        print("Need at least 2 different letters to train a classifier. Collect more data first.")
        return

    counts_per_letter = {label: int((y == label).sum()) for label in unique_labels}
    thin_letters = [label for label, count in counts_per_letter.items() if count < 5]
    if thin_letters:
        print(f"Warning: very few samples for {thin_letters} -- accuracy for these will be unreliable.")

    # stratify keeps the train/test split's letter proportions matching
    # the full dataset; falls back to a plain split if any letter has
    # too few samples for stratification to work.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        print("Too few samples for some letters to stratify the split -- using a plain random split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nMeasured test accuracy: {accuracy:.1%} (on {len(X_test)} held-out samples)")
    print("This is the real, measured number -- never assume it's higher than this.\n")
    print("Per-letter precision/recall/f1:")
    print(classification_report(y_test, y_pred, zero_division=0))

    ASL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, ASL_MODEL_PATH)
    print(f"Saved model to {ASL_MODEL_PATH}")
    print("Restart the Streamlit app (or rerun it) to pick up the new model.")


if __name__ == "__main__":
    main()
