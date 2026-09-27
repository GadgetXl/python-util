from scapy.all import rdpcap, TCP, Raw, IP
from Crypto.Cipher import AES
import hashlib
import sys

if len(sys.argv) < 2:
    print("Usage: python decrypt_captured_packets.py CAPTURE_FILE")
    sys.exit(1)

pcap = rdpcap(sys.argv[1])
target_ip = "TARGET_IP"

# AES KEY CALCULATION
password = "AES_KEY"
data = password.encode("utf-16le")
sha = hashlib.sha256(data).digest()
key = sha[:16]
iv = bytes(16)

# Start and end bytes
START_MARKER = b"" # ex. \xff\xff\xff
END_MARKER = b""

def decrypt(data):
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plain = cipher.decrypt(data)
    pad = plain[-1]
    if pad <= 16:
        plain = plain[:-pad]
    return plain

server_file = open("server_to_client.txt", "w")
client_file = open("client_to_server.txt", "w")

for pkt in pcap:
    if not pkt.haslayer(IP) or not pkt.haslayer(TCP) or not pkt.haslayer(Raw):
        continue

    payload = bytes(pkt[Raw].load)
    if len(payload) == 0:
        continue

    ip = pkt[IP]

    if ip.src == target_ip:
        file = server_file
    elif ip.dst == target_ip:
        file = client_file
    else:
        continue

    # Search for start and end bytes
    start_idx = payload.find(START_MARKER)
    end_idx = payload.find(END_MARKER)

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        continue

    header = payload[:start_idx + len(START_MARKER)]
    encrypted = payload[start_idx + len(START_MARKER):end_idx]
    frame = payload[end_idx:]

    if len(encrypted) == 0 or len(encrypted) % 16 != 0:
        continue

    try:
        plain = decrypt(encrypted)
        result = header + plain + frame
        file.write(result.hex(" ") + "\n\n")
    except Exception as e:
        print("Decrypt error:", e)

server_file.close()
client_file.close()

print("Finished")