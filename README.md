Network Traffic Anomaly Detection
Detect anomalies within network traffic through sophisticated machine learning techniques. Utilizing the power of packet sniffing combined with the Isolation Forest algorithm, this project aims to uncover hidden threats and ensure network integrity.

About The Project
In the vast streams of network traffic, anomalies can signify various issues, from security breaches to system malfunctions. This project seeks to automate the detection of such anomalies, providing a first line of defense against potential threats. By analyzing captured network packets and employing machine learning, we strive to identify unusual patterns indicative of underlying problems.

Built With
Python 3: The primary programming language used.
Scapy: For packet capturing and manipulation.
Pandas & NumPy: For data manipulation and numerical operations.
Scikit-learn: For implementing the Isolation Forest algorithm.
Getting Started
To get a local copy up and running follow these simple steps.

Prerequisites
Ensure you have Python 3 installed on your system. The following additional libraries are required:
scapy
pandas
numpy
scikit-learn

Installation

Clone the repository:
git clone https://github.com/yourusername/network-traffic-anomaly-detection.git
Install Python packages:
pip install -r requirements.txt
Usage
Follow these steps to capture network packets, preprocess the data, and run the anomaly detection model:

Capture Network Packets:


Copy code
python packet_sniffer.py
Adjust the script to specify the network interface and other parameters as needed.

Data Preprocessing:
Process the captured data to format it suitably for machine learning analysis.

Depending which model you want, run the according file:
For Anomaly Detection run the Isolation Forest Model which is as name: isolationforestmodel.py
For Network analysis and Classification run the Random Classifier Model file: randomclassifiermodel.py


Contributing
Contributions make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Fork the Project
Create your Feature Branch (git checkout -b feature/AmazingFeature)
Commit your Changes (git commit -m 'Add some AmazingFeature')
Push to the Branch (git push origin feature/AmazingFeature)
Open a Pull Request
License
Distributed under the MIT License. See LICENSE for more information.

