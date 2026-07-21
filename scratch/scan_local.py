# scratch/scan_local.py
import socket
import threading

found = []
lock = threading.Lock()

def check_port(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.1)
    try:
        s.connect(('127.0.0.1', port))
        with lock:
            print(f"FOUND OPEN LOCAL PORT: {port}", flush=True)
            found.append(port)
        s.close()
    except Exception:
        pass

print("Scanning localhost (127.0.0.1) for open ports in range 500-35000...", flush=True)
threads = []
for port in range(500, 35000):
    t = threading.Thread(target=check_port, args=(port,))
    threads.append(t)
    t.start()
    
    # Simple rate limiting for resource protection
    if len(threads) >= 1000:
        for th in threads:
            th.join()
        threads = []

for th in threads:
    th.join()

print("\n--- Scan Results ---", flush=True)
if found:
    print("Open local ports:", found, flush=True)
else:
    print("No open local ports found in the 35000-50000 range.", flush=True)
