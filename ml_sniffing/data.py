"""Data loading and persistence helpers for packet captures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "source_ip", "destination_ip", "protocol", "length"]


@dataclass(frozen=True)
class PacketData:
    """Container for a dataframe representing captured packets."""

    dataframe: pd.DataFrame

    def ensure_columns(self) -> None:
        missing = [column for column in REQUIRED_COLUMNS if column not in self.dataframe.columns]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"The capture is missing the following required columns: {missing_str}")


def load_packets(path: Path | str) -> PacketData:
    """Load packets from ``path`` into a :class:`PacketData` wrapper."""

    capture_path = Path(path)
    if not capture_path.exists():
        raise FileNotFoundError(f"Could not find packet capture CSV at {capture_path}")

    dataframe = pd.read_csv(capture_path)
    packet_data = PacketData(dataframe=dataframe)
    packet_data.ensure_columns()
    return packet_data


def save_dataframe(dataframe: pd.DataFrame, path: Path | str) -> Path:
    """Persist ``dataframe`` to disk and return the path used."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path


def ensure_non_empty(records: Iterable[dict]) -> List[dict]:
    """Validate that at least one record has been captured."""

    as_list = list(records)
    if not as_list:
        raise ValueError("No packets were captured. Try increasing the capture duration or packet count.")
    return as_list
