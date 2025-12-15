from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------
# CACHE FILES (produced by your feature-building script)
# -----------------------------
CACHE_DIR = Path("cache_features")
TRAIN_CACHE = CACHE_DIR / "train_hog_64.npz"
TEST_CACHE  = CACHE_DIR / "test_hog_64.npz"


def load_cached(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing cache file: {npz_path.resolve()}\n"
            f"Run your feature-building script first to create it."
        )
    data = np.load(npz_path)
    return data["X"], data["y"]


def eval_model(name: str, model, X_test, y_test):
    print(f"\n=== {name} ===")
    t0 = time.time()
    y_pred = model.predict(X_test)
    t_pred = time.time() - t0

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Empty", "Occupied"],
            digits=4
        )
    )
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Prediction time: {t_pred:.2f} sec")



def main():
    print("Loading cached features...")
    X_train, y_train = load_cached(TRAIN_CACHE)
    X_test, y_test   = load_cached(TEST_CACHE)

    print("\nShapes:")
    print("TRAIN:", X_train.shape, "Empty:", int(np.sum(y_train == 0)), "Occ:", int(np.sum(y_train == 1)))
    print("TEST: ", X_test.shape,  "Empty:", int(np.sum(y_test == 0)),  "Occ:", int(np.sum(y_test == 1)))

    # -----------------------------
    # Standardize (fit on train ONLY)
    # -----------------------------
    print("\nStandardizing features...")
    scaler = StandardScaler()
    t0 = time.time()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    print(f"Scaling time: {time.time() - t0:.2f} sec")

    # -----------------------------
    # Model 1: Logistic Regression
    # -----------------------------
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1
    )
    t0 = time.time()
    lr.fit(X_train_s, y_train)
    print(f"LR training time: {time.time() - t0:.2f} sec")
    eval_model("Logistic Regression", lr, X_test_s, y_test)

    # -----------------------------
    # Model 2: Linear SVM
    # -----------------------------
    # LinearSVC is a scalable linear max-margin classifier.
    # Note: It does NOT output probabilities by default.
    print("\nTraining Linear SVM (LinearSVC)...")
    svm = LinearSVC(
        class_weight="balanced",
        dual="auto",
        max_iter=5000
    )
    t0 = time.time()
    svm.fit(X_train_s, y_train)
    print(f"Linear SVM training time: {time.time() - t0:.2f} sec")
    eval_model("Linear SVM (LinearSVC)", svm, X_test_s, y_test)


if __name__ == "__main__":
    main()
