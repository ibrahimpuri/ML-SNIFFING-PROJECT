from scapy.all import sniff, IP
import pandas as pd

# Replace 'en0' with the interface you identified using ifconfig
INTERFACE = 'eno'

packets_data = []  # List to store packet info

def packet_callback(packet):
    """Callback function to process each packet"""
    if IP in packet:  # Check if it's an IP packet
        packet_info = {
            'timestamp': packet.time,
            'source_ip': packet[IP].src,
            'destination_ip': packet[IP].dst,
            'protocol': packet.sprintf("%IP.proto%"),
            'length': len(packet)
        }
        packets_data.append(packet_info)

def start_sniffing(interface=INTERFACE, packet_count=100):
    """Start packet sniffing on the specified interface"""
    print(f"Starting packet sniffing on {interface}... (Press CTRL+C to stop)")
    sniff(iface=interface, prn=packet_callback, count=packet_count)

def save_packets_to_csv(file_name='captured_packets.csv'):
    """Save captured packets to a CSV file"""
    df_packets = pd.DataFrame(packets_data)
    df_packets.to_csv(file_name, index=False)
    print(f"Saved captured packets to {file_name}")

if __name__ == "__main__":
    start_sniffing()
    save_packets_to_csv()