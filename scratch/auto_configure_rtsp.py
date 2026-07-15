# scratch/auto_configure_rtsp.py
import socket
import cv2
import re
import os
from concurrent.futures import ThreadPoolExecutor

targets = ["127.0.0.1", "192.168.1.7", "192.168.1.252"]
username = "admin"
password = "sheild222@"
found_ip = None
found_port = None

def check_port_and_stream(ip, port):
    global found_ip, found_port
    if found_ip is not None:
        return
        
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.15)
    try:
        s.connect((ip, port))
        s.close()
        # Port is open, test RTSP stream
        url = f"rtsp://{username}:sheild222@@{ip}:{port}/cam/realmonitor?channel=3&subtype=0"
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            found_ip = ip
            found_port = port
            print(f"\n>>> SUCCESS: Found working RTSP stream at {ip}:{port}!", flush=True)
            cap.release()
    except Exception:
        pass

print("Scanning for working RTSP proxy ports in range 30000-50000...", flush=True)
with ThreadPoolExecutor(max_workers=300) as executor:
    for ip in targets:
        if found_ip is not None:
            break
        print(f"Probing {ip}...", flush=True)
        futures = [executor.submit(check_port_and_stream, ip, port) for port in range(30000, 50000)]
        for f in futures:
            if found_ip is not None:
                break

if found_ip and found_port:
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    print(f"Updating {env_path}...", flush=True)
    with open(env_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r"RTSP_URL=.*", "RTSP_URL=", content)
    content = re.sub(r"RTSP_HOST=.*", f"RTSP_HOST={found_ip}", content)
    content = re.sub(r"RTSP_PORT=.*", f"RTSP_PORT={found_port}", content)

    with open(env_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("\n=======================================================", flush=True)
    print("  AUTO-CONFIGURATION SUCCESSFUL!", flush=True)
    print(f"  Updated .env to use:")
    print(f"  RTSP_HOST = {found_ip}")
    print(f"  RTSP_PORT = {found_port}")
    print("  Please restart main.py to see the live stream dashboard!", flush=True)
    print("=======================================================", flush=True)
else:
    print("\n[FAIL] No active RTSP proxy stream found in the 30000-50000 range.", flush=True)
    print("Please make sure SmartPSS Lite is running and the camera Live View is OPEN.", flush=True)
