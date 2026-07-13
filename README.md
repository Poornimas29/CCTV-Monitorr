# AI Employee Monitoring System - Phase 1: RTSP Stream Viewer

This repository contains Phase 1 of the **AI Employee Monitoring System**. The goal of this phase is to establish a robust, multithreaded connection to a Dahua RTSP stream, validate that the feed is functional, and display the live video with a diagnostic overlay in real time.

It is written in Python, uses OpenCV and NumPy, and features clean, Object-Oriented design, auto-reconnection, and keyboard shortcuts.

---

## Features

- **Multithreaded Frame Capture:** Frame reading runs on a dedicated background thread, ensuring smooth GUI displaying and eliminating frame buffering lag.
- **Auto-Reconnection:** If the RTSP stream drops, the background thread automatically triggers a retry loop every 5 seconds without crashing the application.
- **Diagnostic Overlay:** Draws a sleek semi-transparent control panel showing Camera Status, stream input FPS, source Resolution, and Current Date & Time.
- **Testing Mock Mode:** Runs a synthetic video simulation (radar Sweep and targets) if `RTSP_URL` is set to `mock`. This allows developers to test the viewer without having a physical Dahua camera connected.
- **Keyboard Shortcuts:**
  - `S`: Save a raw JPG snapshot in the `captures/` folder with filename format `capture_YYYYMMDD_HHMMSS.jpg`
  - `Q`: Gracefully disconnect threads, release OpenCV resources, and close the application.
- **Robust Logger:** Logs all events (startup, connecting, connection successful, connection lost, reconnect retries, snapshot saved, and shutdown) to both console and `logs/app.log`.

---

## Directory Structure

```text
employee_monitoring/
│
├── stream/
│   ├── __init__.py
│   ├── rtsp_stream.py          # Multithreaded reader thread with Mock Mode
│   ├── stream_manager.py       # Wrapper API coordinating reader thread & buffer
│   ├── reconnect_handler.py    # Tracks disconnections & schedules retry wait time
│   └── frame_buffer.py         # Thread-safe buffer keeping only the latest frame
│
├── config/
│   ├── __init__.py
│   ├── settings.py             # Loads and parses environment configuration
│   └── logging_config.py       # Configures root logging handlers for file & console
│
├── logs/
│   └── app.log                 # Log outputs (created dynamically)
│
├── captures/
│   └── capture_*.jpg           # Snapshots taken using the 'S' key (created dynamically)
│
├── tests/
│   ├── __init__.py
│   └── test_stream_components.py # Unit tests for config, buffer, and reconnect logic
│
├── main.py                     # Entry point containing the display loop
├── requirements.txt            # Package dependencies
├── .env.example                # Configuration template
├── .env                        # Live local environment settings file (ignored in git)
└── README.md                   # Setup and usage guide
```

---

## Setup Instructions

### 1. Prerequisites
Ensure you have **Python 3.11+** installed on your system.

### 2. Install Dependencies
Open your terminal in the project root directory and run:
```bash
pip install -r requirements.txt
```

### 3. Configure the Environment
Copy the example environment file to `.env`:
```bash
copy .env.example .env
```
*(On Linux/macOS: `cp .env.example .env`)*

Open the `.env` file and set the RTSP stream URL.
- **For Testing (Mock Mode):**
  ```env
  RTSP_URL=mock
  ```
- **For Production (Dahua RTSP Camera):**
  ```env
  RTSP_URL=rtsp://username:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0
  ```

---

## Running the Application

To run the live stream viewer:
```bash
python main.py
```

### Key Controls
While the viewer window is focused:
- **`S`**: Saves a clean screenshot of the camera frame (without the status bar overlays) to the `captures/` folder. A notification will flash on screen.
- **`Q`** (or closing the window): Gracefully shuts down reader threads and exits.

---

## Running Unit Tests

The test suite validates the internal mechanics (such as the thread-safe frame buffer, reconnect timers, and settings loading):
```bash
python -m unittest discover -s tests
```
