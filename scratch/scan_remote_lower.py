# scratch/scan_remote_lower.py
import socket
import threading

target_ip = "192.168.1.7"
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

print(f"Scanning remote host {target_ip} for open ports in range 1000-35000...", flush=True)
threads = []
for port in range(1000, 35000):
    t = threading.Thread(target=check_port, args=(port,))
    threads.append(t)
    t.start()
    
    # Batch joining to avoid thread overload
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
    print(f"No open ports found on {target_ip} in the 1000-35000 range.", flush=True)
