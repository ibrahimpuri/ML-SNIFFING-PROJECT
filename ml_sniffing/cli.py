"""Command line interface for the ML Sniffing project."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from . import classification, data, features, isolation, sniffer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tools for capturing and analysing network packets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sniff_parser = subparsers.add_parser("sniff", help="Capture packets using Scapy")
    sniff_parser.add_argument("output", type=Path, help="Destination CSV file for the captured packets")
    sniff_parser.add_argument("--interface", "-i", required=True, help="Network interface to sniff")
    sniff_parser.add_argument(
        "--packet-count",
        "-c",
        type=int,
        default=None,
        help="Number of packets to capture (omit for continuous capture)",
    )

    detect_parser = subparsers.add_parser("detect", help="Run Isolation Forest anomaly detection")
    detect_parser.add_argument("input", type=Path, help="CSV file containing captured packets")
    detect_parser.add_argument("--output", type=Path, help="Optional CSV to store anomaly scores")
    detect_parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Expected proportion of anomalies in the data (0, 0.5]",
    )
    detect_parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of most anomalous packets to display",
    )

    classify_parser = subparsers.add_parser("classify", help="Train a Random Forest classifier on labelled data")
    classify_parser.add_argument("input", type=Path, help="CSV file containing captured packets")
    classify_parser.add_argument(
        "--label-column",
        required=True,
        help="Name of the column containing the ground-truth labels",
    )
    classify_parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of samples to keep for evaluation",
    )
    classify_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for the train/test split and model initialisation",
    )

    return parser


def _handle_sniff(args: argparse.Namespace) -> int:
    config = sniffer.CaptureConfig(
        interface=args.interface,
        packet_count=args.packet_count,
        output=args.output,
    )
    output_path = sniffer.capture_and_save(config)
    print(f"Captured packets saved to {output_path}")
    return 0


def _handle_detect(args: argparse.Namespace) -> int:
    capture = data.load_packets(args.input)
    feature_set = features.engineer_features(capture.dataframe)
    result = isolation.detect_anomalies(feature_set.features, contamination=args.contamination)

    augmented = feature_set.augmented.join(result.anomaly_scores).join(result.is_anomaly)
    augmented.sort_values("anomaly_score", inplace=True)

    if args.output:
        output_path = data.save_dataframe(augmented, args.output)
        print(f"Anomaly scores written to {output_path}")

    summary = isolation.summarise_anomalies(result, top_n=args.top_n)
    print("Most anomalous packets:")
    print(augmented.loc[summary.index, :].head(args.top_n))
    return 0


def _handle_classify(args: argparse.Namespace) -> int:
    capture = data.load_packets(args.input)
    if args.label_column not in capture.dataframe.columns:
        raise ValueError(
            f"Column '{args.label_column}' was not found in {args.input}. "
            "Provide a labelled dataset or choose a different column."
        )

    feature_set = features.engineer_features(capture.dataframe)
    target = capture.dataframe[args.label_column]

    result = classification.train_random_forest(
        feature_set.features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("Evaluation metrics:")
    for metric, value in result.metrics.items():
        print(f"  {metric}: {value:.3f}")

    print("\nDetailed classification report:")
    print(result.report)

    print("\nConfusion matrix:")
    print(result.confusion_matrix)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "sniff":
            return _handle_sniff(args)
        if args.command == "detect":
            return _handle_detect(args)
        if args.command == "classify":
            return _handle_classify(args)
    except Exception as exc:  # pragma: no cover - CLI safety net
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
