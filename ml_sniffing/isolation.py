"""Isolation Forest based anomaly detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass(frozen=True)
class IsolationForestResult:
    """Result bundle from running the Isolation Forest detector."""

    model: IsolationForest
    anomaly_scores: pd.Series
    is_anomaly: pd.Series


def detect_anomalies(
    features: pd.DataFrame,
    contamination: float = 0.05,
    n_estimators: int = 200,
    random_state: Optional[int] = 42,
) -> IsolationForestResult:
    """Fit an Isolation Forest and produce anomaly labels and scores."""

    if not 0 < contamination <= 0.5:
        raise ValueError("contamination must be in the range (0, 0.5]")

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(features)

    scores = model.decision_function(features)
    labels = model.predict(features)

    result = IsolationForestResult(
        model=model,
        anomaly_scores=pd.Series(scores, index=features.index, name="anomaly_score"),
        is_anomaly=pd.Series(labels == -1, index=features.index, name="is_anomaly"),
    )
    return result


def summarise_anomalies(result: IsolationForestResult, top_n: int = 10) -> pd.DataFrame:
    """Return the ``top_n`` most anomalous records sorted by score ascending."""

    top_n = max(int(top_n), 1)
    summary = (
        pd.concat([result.anomaly_scores, result.is_anomaly], axis=1)
        .sort_values("anomaly_score")
        .head(top_n)
    )
    return summary
