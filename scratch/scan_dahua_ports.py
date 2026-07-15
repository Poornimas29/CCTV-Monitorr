# scratch/scan_dahua_ports.py
import socket
import threading

target_ip = "192.168.1.206"
found = []
lock = threading.Lock()

def check_port(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.2)
    try:
        s.connect((target_ip, port))
        with lock:
            print(f"FOUND OPEN PORT ON {target_ip}: {port}", flush=True)
            found.append(port)
        s.close()
    except Exception:
        pass

print(f"Scanning remote Dahua IP {target_ip} for open ports in range 30000-50000...", flush=True)
threads = []
for port in range(30000, 50000):
    t = threading.Thread(target=check_port, args=(port,))
    threads.append(t)
    t.start()
    
    # Batch joining to avoid thread/socket overload
    if len(threads) >= 1000:
        for th in threads:
            th.join()
        threads = []

for th in threads:
    th.join()

print("\n--- Scan Results ---", flush=True)
if found:
    print(f"Open ports on {target_ip}:", found, flush=True)
else:
    print(f"No open ports found on {target_ip} in the 30000-50000 range.", flush=True)
