import pandas as pd

# Load the dataset
df = pd.read_csv('captured_packets.csv')

# Display the first few rows of the dataset
print(df.head())

# Get a concise summary of the DataFrame
print(df.info())

# Basic statistics for numerical columns
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Histogram of packet lengths
df['length'].hist(bins=50)

# Count of packets by protocol
print(df['protocol'].value_counts())

# Boxplot for packet length can help identify outliers
df.boxplot(column=['length'])

import pandas as pd

# Assuming 'timestamp' is in UNIX time format
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# Extract time-based features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6

# Optionally, you can create a feature for the time of day (morning, afternoon, evening)
df['time_of_day'] = pd.cut(df['hour'], 
                           bins=[0, 6, 12, 18, 24], 
                           include_lowest=True, 
                           labels=['Night', 'Morning', 'Afternoon', 'Evening'])

# Function to check if an IP address is private
def is_private_ip(ip):
    if ip.startswith('10.') or ip.startswith('192.168.') or ip.startswith('172.'):
        return 'Private'
    else:
        return 'Public'

# Apply the function to source and destination IP addresses
df['source_ip_type'] = df['source_ip'].apply(is_private_ip)
df['destination_ip_type'] = df['destination_ip'].apply(is_private_ip)

# Create a unique identifier for each flow based on IP addresses
df['flow_id'] = df.apply(lambda row: '_'.join(sorted([row['source_ip'], row['destination_ip']])), axis=1)

# Count packets in each flow
df['flow_packet_count'] = df.groupby('flow_id')['timestamp'].transform('count')

# Average packet size in each flow
df['flow_avg_packet_size'] = df.groupby('flow_id')['length'].transform('mean')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

# Assuming df is your DataFrame, and it's ready for modeling
# Note: Anomaly detection is often unsupervised, so we may not split into X and y

# Isolation Forest expects the anomaly class to be -1 and the normal class to be 1
# If your dataset is labeled, ensure the target column is adjusted accordingly

X_train, X_test = train_test_split(df, test_size=0.3, random_state=42)

# Initialize Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
iso_forest.fit(X_train)

# Predictions (anomaly scores)
scores = iso_forest.decision_function(X_test)
# Anomalies are labeled as -1
labels = iso_forest.predict(X_test)

# You can then analyze the scores and labels to identify anomalies