"""Supervised classification utilities for labelled packet datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ClassificationResult:
    """Encapsulates the outcome of a classification training run."""

    model: RandomForestClassifier
    metrics: Dict[str, float]
    report: pd.DataFrame
    confusion_matrix: np.ndarray


def train_random_forest(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.3,
    random_state: Optional[int] = 42,
    n_estimators: int = 200,
) -> ClassificationResult:
    """Train a Random Forest classifier and return evaluation metrics."""

    if target.nunique() < 2:
        raise ValueError("The target column must contain at least two distinct classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": report_dict.get("accuracy", 0.0),
        "macro_precision": report_dict.get("macro avg", {}).get("precision", 0.0),
        "macro_recall": report_dict.get("macro avg", {}).get("recall", 0.0),
        "macro_f1": report_dict.get("macro avg", {}).get("f1-score", 0.0),
    }

    report = pd.DataFrame(report_dict).transpose()
    cmatrix = confusion_matrix(y_test, y_pred)

    return ClassificationResult(model=model, metrics=metrics, report=report, confusion_matrix=cmatrix)
