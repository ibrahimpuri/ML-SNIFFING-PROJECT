"""Packet sniffing utilities built on top of Scapy."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:  # pragma: no cover - optional dependency is validated at runtime
    from scapy.all import IP, sniff
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "scapy is required for packet sniffing. Install it with 'pip install scapy'."
    ) from exc

from . import data


@dataclass(frozen=True)
class CaptureConfig:
    """Configuration for a packet capture run."""

    interface: str
    packet_count: Optional[int]
    output: Path


def _packet_to_record(packet) -> Optional[dict]:  # type: ignore[no-untyped-def]
    """Extract relevant information from a Scapy packet."""

    if IP not in packet:
        return None

    return {
        "timestamp": float(packet.time),
        "source_ip": packet[IP].src,
        "destination_ip": packet[IP].dst,
        "protocol": packet.sprintf("%IP.proto%"),
        "length": int(len(packet)),
    }


def capture_packets(config: CaptureConfig) -> pd.DataFrame:
    """Run a packet capture according to ``config`` and return a dataframe."""

    records: List[dict] = []

    def _callback(packet) -> None:  # type: ignore[no-untyped-def]
        record = _packet_to_record(packet)
        if record:
            records.append(record)

    sniff(
        iface=config.interface,
        prn=_callback,
        count=config.packet_count,
        store=False,
    )

    validated_records = data.ensure_non_empty(records)
    return pd.DataFrame(validated_records)


def capture_and_save(config: CaptureConfig) -> Path:
    """Capture packets and persist them to ``config.output``."""

    dataframe = capture_packets(config)
    return data.save_dataframe(dataframe, config.output)
