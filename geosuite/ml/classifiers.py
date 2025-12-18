from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


@dataclass
class FaciesResult:
    classes_: List[str]
    y_pred: pd.Series
    proba: pd.DataFrame  # columns = classes_, rows aligned to input index
    model_name: str
    report: str


def train_and_predict(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_type: str = "SVM",
    test_size: float = 0.0,
    random_state: int = 42,
) -> FaciesResult:
    """
    Train a classifier (SVM with probability, or RandomForest) and predict on the full dataset.
    When test_size > 0, produces a report on the holdout set but still returns predictions on all rows for plotting.
    """
    X = df[feature_cols].copy()
    y = df[target_col].astype(str).copy()

    if model_type.upper() == "SVM":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=random_state)),
        ])
        model_name = "SVM (RBF)"
    elif model_type.upper() in ("RF", "RANDOMFOREST", "RANDOM_FOREST"):
        pipe = Pipeline([
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=None, random_state=random_state))
        ])
        model_name = "RandomForest"
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    report = ""
    if test_size and test_size > 0:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        pipe.fit(Xtr, ytr)
        yhat_te = pipe.predict(Xte)
        report = classification_report(yte, yhat_te)
    else:
        pipe.fit(X, y)

    # Predict on the full dataset for visualization consistency
    y_pred = pd.Series(pipe.predict(X), index=df.index, name="predicted")
    if hasattr(pipe, "predict_proba"):
        P = pipe.predict_proba(X)
        classes = list(pipe.classes_)
    else:
        # Shouldn't happen for the two models above, but keep a fallback
        # Use one-vs-rest decision function like probabilities
        if hasattr(pipe, "decision_function"):
            S = pipe.decision_function(X)
            # softmax-like normalization
            expS = np.exp(S - np.max(S, axis=1, keepdims=True))
            P = expS / np.sum(expS, axis=1, keepdims=True)
            classes = list(np.unique(y))
        else:
            # No scores; default to 1.0 for predicted class
            classes = list(np.unique(y))
            P = np.zeros((len(X), len(classes)))
            class_to_idx = {c: i for i, c in enumerate(classes)}
            for i, c in enumerate(y_pred):
                P[i, class_to_idx[c]] = 1.0

    proba = pd.DataFrame(P, index=df.index, columns=classes)

    return FaciesResult(
        classes_=classes,
        y_pred=y_pred,
        proba=proba,
        model_name=model_name,
        report=report,
    )
