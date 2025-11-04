"""Feature engineering utilities for packet captures."""
from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_address, ip_network
from typing import Iterable

import numpy as np
import pandas as pd

PRIVATE_NETWORKS = (
    ip_network("10.0.0.0/8"),
    ip_network("172.16.0.0/12"),
    ip_network("192.168.0.0/16"),
)


@dataclass(frozen=True)
class FeatureSet:
    """Container bundling engineered features with the augmented source data."""

    features: pd.DataFrame
    augmented: pd.DataFrame


def _ip_to_int(ip: str) -> int:
    """Convert an IPv4 or IPv6 string into a deterministic integer."""

    try:
        return int(ip_address(ip))
    except ValueError:
        return 0


def _is_private(ip: str) -> bool:
    try:
        ip_obj = ip_address(ip)
    except ValueError:
        return False
    return any(ip_obj in network for network in PRIVATE_NETWORKS)


def engineer_features(dataframe: pd.DataFrame) -> FeatureSet:
    """Generate model-friendly features from a raw capture dataframe."""

    df = dataframe.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    df["hour"] = df["timestamp"].dt.hour.fillna(0).astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)
    df["minute"] = df["timestamp"].dt.minute.fillna(0).astype(int)

    df["source_ip_int"] = df["source_ip"].apply(_ip_to_int)
    df["destination_ip_int"] = df["destination_ip"].apply(_ip_to_int)
    df["source_is_private"] = df["source_ip"].apply(_is_private).astype(int)
    df["destination_is_private"] = df["destination_ip"].apply(_is_private).astype(int)

    df["protocol"] = df["protocol"].astype(str).str.lower()
    df["protocol_code"] = df["protocol"].astype("category").cat.codes.astype(int)

    flow_id = (
        df["source_ip"].astype(str)
        + "->"
        + df["destination_ip"].astype(str)
        + ":"
        + df["protocol"].astype(str)
    )
    df["flow_id"] = flow_id

    flow_stats = df.groupby("flow_id")["length"].agg(["count", "mean"]).rename(
        columns={"count": "flow_packet_count", "mean": "flow_avg_length"}
    )
    df = df.join(flow_stats, on="flow_id")

    df["flow_packet_count"] = df["flow_packet_count"].fillna(1)
    df["flow_avg_length"] = df["flow_avg_length"].fillna(df["length"])

    numeric_columns = [
        "length",
        "hour",
        "day_of_week",
        "minute",
        "source_ip_int",
        "destination_ip_int",
        "source_is_private",
        "destination_is_private",
        "protocol_code",
        "flow_packet_count",
        "flow_avg_length",
    ]

    features = df[numeric_columns].astype(np.float64)
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)

    return FeatureSet(features=features, augmented=df)


def select_features(feature_set: FeatureSet, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Retrieve a subset of feature columns while preserving the original index."""

    if columns is None:
        return feature_set.features
    missing = [column for column in columns if column not in feature_set.features.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"The following feature columns are unavailable: {missing_str}")
    return feature_set.features.loc[:, list(columns)]
