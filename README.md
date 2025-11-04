# Network Traffic Anomaly Detection

Detect anomalies within network traffic through machine learning techniques. This project provides tooling to capture packets,
engineer features and run both unsupervised anomaly detection and supervised classification experiments.

## Getting Started

### Prerequisites

Ensure you have Python 3.9 or later installed. Install project dependencies using pip:

```bash
pip install -r requirements.txt
```

### Capturing Packets

Packet capture requires root/administrator privileges and an available network interface. Use the CLI to capture packets to a
CSV file:

```bash
python -m ml_sniffing.cli sniff captured_packets.csv --interface eth0 --packet-count 500
```

Omit `--packet-count` to keep sniffing until interrupted with `CTRL+C`.

### Detecting Anomalies

Run the Isolation Forest detector on a CSV of captured packets:

```bash
python -m ml_sniffing.cli detect captured_packets.csv --top-n 5 --output anomalies.csv
```

The command prints the most anomalous packets and optionally stores all anomaly scores in `anomalies.csv`.

### Training a Classifier

If you have a labelled dataset (for example with a column named `is_malicious`), train and evaluate a Random Forest classifier:

```bash
python -m ml_sniffing.cli classify captured_packets.csv --label-column is_malicious
```

The script prints accuracy, macro-averaged metrics, a detailed classification report and the confusion matrix.

## Project Structure

- `ml_sniffing/cli.py` – user-facing command line interface
- `ml_sniffing/sniffer.py` – packet capture helpers built on top of Scapy
- `ml_sniffing/data.py` – CSV loading and persistence helpers
- `ml_sniffing/features.py` – feature engineering shared by the models
- `ml_sniffing/isolation.py` – Isolation Forest anomaly detector utilities
- `ml_sniffing/classification.py` – Random Forest classifier utilities

## Contributing

Contributions make the open-source community an amazing place to learn, inspire and create. Any contributions you make are
appreciated.

1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License

Distributed under the MIT License. See `LICENSE` for more information.
