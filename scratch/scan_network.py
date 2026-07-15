# scratch/scan_network.py
import socket
import threading
import sys

subnet = "192.168.1"
ports = [554, 8554, 42945, 48356]
found = []
lock = threading.Lock()

def check_ip(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.4)
    try:
        s.connect((ip, port))
        with lock:
            print(f"FOUND: {ip}:{port} is OPEN", flush=True)
            found.append((ip, port))
        s.close()
    except Exception:
        pass

print(f"Scanning subnet {subnet}.* for ports {ports}...", flush=True)
threads = []
for i in range(1, 255):
    ip = f"{subnet}.{i}"
    for port in ports:
        t = threading.Thread(target=check_ip, args=(ip, port))
        threads.append(t)
        t.start()

for t in threads:
    t.join()

print("\n--- Scan Results ---", flush=True)
if found:
    for ip, port in found:
        print(f"Active Camera Target: RTSP_HOST={ip} and RTSP_PORT={port}", flush=True)
else:
    print("No open RTSP target ports found on the subnet.", flush=True)
