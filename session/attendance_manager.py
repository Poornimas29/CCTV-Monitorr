# session/attendance_manager.py
"""AttendanceManager manages attendance sessions, working hours, and reports.

Output folder structure
-----------------------
output/
  known/    ← one JSON per identified employee session
  unknown/  ← one JSON per unrecognized person track
  reports/  ← daily/weekly summary reports
"""

import os
import glob
import json
import logging
from datetime import datetime
from typing import List, Optional, Any, Tuple
import numpy as np
import config.settings as settings
from session.global_session_manager import GlobalSessionManager, GlobalSession

logger = logging.getLogger(__name__)

# ── Sub-folder helpers ─────────────────────────────────────────────────────────────────
def _subdir(name: str) -> str:
    """Return path of output/<name>/ creating it if necessary."""
    path = os.path.join(settings.OUTPUT_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def known_dir() -> str:   return _subdir("known")
def unknown_dir() -> str: return _subdir("unknown")
def reports_dir() -> str: return _subdir("reports")


class AttendanceManager(GlobalSessionManager):
    """Manages global employee attendance session states, working durations, and logs."""
    
    def __init__(self, lost_timeout_seconds: float = None) -> None:
        timeout = lost_timeout_seconds if lost_timeout_seconds is not None else settings.TRACK_TIMEOUT
        super().__init__(lost_timeout_seconds=int(timeout))

    def create_session(
        self,
        employee_id: str,
        employee_name: str,
        camera_id: str,
        track_id: int,
        bbox: List[int],
        timestamp: datetime,
        confidence: float,
        reid_features: Optional[np.ndarray] = None,
        reid_hist: Optional[np.ndarray] = None
    ) -> GlobalSession:
        """Creates or reactivates a global session, logging the event."""
        session = super().create_session(
            employee_id=employee_id,
            employee_name=employee_name,
            camera_id=camera_id,
            track_id=track_id,
            bbox=bbox,
            timestamp=timestamp,
            confidence=confidence,
            reid_features=reid_features,
            reid_hist=reid_hist
        )
        
        # Log Attendance Started if it is a new session
        if session.first_seen == timestamp:
            print("----------------------")
            print("Attendance Started")
            print(f"Employee ID: {employee_id}")
            print(f"Employee Name: {employee_name}")
            print(f"Entry Time: {timestamp:%Y-%m-%d %H:%M:%S}")
            print("----------------------")
            logger.info("[Logger] Attendance Started - Employee ID %s at %s", employee_id, timestamp)
            
        return session

    def get_daily_attendance_summary(
        self, employee_id: str, date_str: str, current_first_seen: datetime, current_last_seen: datetime
    ) -> Tuple[datetime, datetime, float, float, int]:
        """Scan today's JSON records in output/known/ to build cumulative totals."""
        first_entry = current_first_seen
        last_exit = current_last_seen
        total_work_seconds = 0.0
        total_phone_seconds = 0.0
        total_phone_count = 0
        try:
            pattern = os.path.join(known_dir(), f"attendance_{employee_id}_{date_str}_*.json")
            for filepath in glob.glob(pattern):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    entry_val = datetime.fromisoformat(data["entry_time"])
                    exit_val = datetime.fromisoformat(data["exit_time"])
                    if entry_val < first_entry:
                        first_entry = entry_val
                    if exit_val > last_exit:
                        last_exit = exit_val
                    total_work_seconds += data.get("working_duration_seconds", 0.0)
                    total_phone_seconds += data.get("phone_use_duration_seconds", 0.0)
                    total_phone_count += data.get("phone_use_count", 0)
                except Exception:
                    pass
        except Exception:
            pass
        return first_entry, last_exit, total_work_seconds, total_phone_seconds, total_phone_count

    def process_timeouts(self, timestamp: datetime) -> List[GlobalSession]:
        """Checks for timed out sessions and records exit details."""
        exited_sessions = super().process_timeouts(timestamp)
        for session in exited_sessions:
            duration_sec = session.working_duration
            m, s = divmod(int(duration_sec), 60)
            h, m = divmod(m, 60)
            duration_str = f"{h}h {m}m {s}s"
            
            # Save the attendance record first so it is written to disk
            self.generate_attendance_record(session)
            
            # Calculate today's full summary
            date_str = session.last_seen.strftime('%Y%m%d')
            first_entry, last_exit, total_work_seconds, total_phone_seconds, total_phone_count = self.get_daily_attendance_summary(
                session.employee_id, date_str, session.first_seen, session.last_seen
            )

            # Working time string
            tot_m, tot_s = divmod(int(total_work_seconds), 60)
            tot_h, tot_m = divmod(tot_m, 60)
            total_work_str = f"{tot_h}h {tot_m}m {tot_s}s"
            # Phone usage string
            ph_m, ph_s = divmod(int(total_phone_seconds), 60)
            ph_h, ph_m = divmod(ph_m, 60)
            total_phone_str = f"{ph_h}h {ph_m}m {ph_s}s"
            
            total_duration_str = total_work_str  # keep existing variable for consistency
            
            print("----------------------")
            print("Attendance Completed")
            print(f"Employee ID: {session.employee_id}")
            print(f"Employee Name: {session.employee_name}")
            print(f"First Appearance (Today): {first_entry:%Y-%m-%d %H:%M:%S}")
            print(f"Last Departure (Today): {last_exit:%Y-%m-%d %H:%M:%S}")
            print(f"Working Hours (Session): {duration_str}")
            print(f"Total Working Hours (Today): {total_work_str}")
            print(f"Total Mobile Usage (Today): {total_phone_str} (Count: {total_phone_count})")
            print("----------------------")
            logger.info(
                "[Logger] Attendance Completed - Employee ID %s | First: %s | Last: %s | Cumulative Today: %s",
                session.employee_id, first_entry, last_exit, total_duration_str
            )
            
        return exited_sessions

    def get_employee_total_summary(self, employee_id: str) -> Tuple[float, float, int]:
        """Aggregate totals for one employee across all known/ records."""
        total_work_seconds = 0.0
        total_phone_seconds = 0.0
        total_phone_count = 0
        try:
            pattern = os.path.join(known_dir(), f"attendance_{employee_id}_*.json")
            for filepath in glob.glob(pattern):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    total_work_seconds += data.get("working_duration_seconds", 0.0)
                    total_phone_seconds += data.get("phone_use_duration_seconds", 0.0)
                    total_phone_count += data.get("phone_use_count", 0)
                except Exception:
                    pass
        except Exception:
            pass
        return total_work_seconds / 3600.0, total_phone_seconds, total_phone_count

    def generate_attendance_record(self, session: "GlobalSession") -> None:
        """Write a JSON attendance record for an identified employee to output/known/."""
        try:
            cam_history_out = [
                {
                    "camera_id": e["cam_id"],
                    "entry_time": e["entry_time"].isoformat() if hasattr(e["entry_time"], "isoformat") else str(e["entry_time"]),
                    "exit_time": e["exit_time"].isoformat() if e["exit_time"] and hasattr(e["exit_time"], "isoformat") else None,
                }
                for e in session.camera_history
            ]
            record = {
                "session_id": session.session_id,
                "employee_id": session.employee_id,
                "employee_name": session.employee_name,
                "entry_time": session.first_seen.isoformat(),
                "exit_time": session.last_seen.isoformat(),
                "working_hours": round(session.working_duration / 3600.0, 4),
                "working_duration_seconds": round(session.working_duration, 2),
                "phone_use_duration_seconds": round(session.phone_use_duration, 2),
                "phone_use_count": len(session.phone_use_history),
                "productivity_score": round(session.productivity_score, 2),
                "recognition_confidence": round(session.recognition_confidence, 2),
                "camera_history": cam_history_out,
            }
            filename = f"attendance_{session.employee_id}_{session.last_seen.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(known_dir(), filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            logger.info("[AttendanceManager] Known record written → %s", filepath)
        except Exception as exc:
            logger.error("[AttendanceManager] Failed to write known record: %s", exc)

    def generate_unrecognized_attendance_record(self, track: dict) -> None:
        """Write a JSON record for an unrecognized track to output/unknown/.

        Auto-clear policy
        -----------------
        Unknown track files are automatically purged after UNKNOWN_TRACK_CLEANUP_MINUTES
        (default 15 minutes) once they are no longer active on screen. This value is set
        in .env as UNKNOWN_TRACK_CLEANUP_MINUTES. The file itself records the cleanup time
        so the client always knows when the record will expire.

        Mobile usage is tracked for unknown persons the same way as for known employees.
        If the person was seen holding a phone, phone_use_duration_seconds > 0.
        """
        try:
            track_id = track.get("track_id", "?")
            entry_time = track.get("entry_time")
            exit_time = track.get("exit_time") or track.get("last_seen")
            camera_id = track.get("camera_id", "")

            # Duration on screen
            dur_sec = 0.0
            if entry_time and exit_time:
                try:
                    dur_sec = (exit_time - entry_time).total_seconds()
                except Exception:
                    pass

            # Phone usage from track memory (may or may not be present)
            phone_sec = float(track.get("phone_use_duration", 0.0))
            phone_cnt = int(track.get("phone_use_count", 0))

            cleanup_minutes = float(getattr(settings, "UNKNOWN_TRACK_CLEANUP_MINUTES", 15.0))

            record = {
                "track_id": track_id,
                "employee_id": None,
                "employee_name": "Unknown",
                "camera_id": camera_id,
                "entry_time": entry_time.isoformat() if hasattr(entry_time, "isoformat") else str(entry_time or ""),
                "exit_time": exit_time.isoformat() if hasattr(exit_time, "isoformat") else str(exit_time or ""),
                "working_hours": round(dur_sec / 3600.0, 6),
                "working_duration_seconds": round(dur_sec, 2),
                "phone_use_duration_seconds": round(phone_sec, 2),
                "phone_use_count": phone_cnt,
                "track_status": "exited",
                "auto_clear_after_minutes": cleanup_minutes,
                "note": f"Record auto-purged from system after {cleanup_minutes:.0f} min of inactivity",
            }

            ts_str = (exit_time or datetime.now()).strftime("%Y%m%d_%H%M%S") if exit_time else datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"unknown_track_{track_id}_{ts_str}.json"
            filepath = os.path.join(unknown_dir(), filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            logger.info("[AttendanceManager] Unknown record written → %s", filepath)
            
            # Clean up expired unknown files to enforce the auto-clear policy on disk
            self.cleanup_expired_unknown_files()
        except Exception as exc:
            logger.error("[AttendanceManager] Failed to write unknown record: %s", exc)

    def cleanup_expired_unknown_files(self) -> None:
        """Scan output/unknown/ and delete any JSON files older than UNKNOWN_TRACK_CLEANUP_MINUTES."""
        try:
            import time
            cleanup_minutes = float(getattr(settings, "UNKNOWN_TRACK_CLEANUP_MINUTES", 15.0))
            now = time.time()
            pattern = os.path.join(unknown_dir(), "unknown_track_*.json")
            for filepath in glob.glob(pattern):
                try:
                    mtime = os.path.getmtime(filepath)
                    if (now - mtime) > cleanup_minutes * 60:
                        os.remove(filepath)
                        logger.info("[AttendanceManager] Auto-cleared expired unknown record: %s", filepath)
                except Exception as e:
                    logger.error("[AttendanceManager] Error cleaning up file %s: %s", filepath, e)
        except Exception as exc:
            logger.error("[AttendanceManager] Error during unknown files cleanup: %s", exc)


    def generate_daily_summary_report(self, date_str: str) -> None:
        """Generate the End-of-Day (EOD) report with three partitions.

        Saved files
        -----------
        output/reports/daily_report_<YYYYMMDD>.json   — combined master report (3 sections)
        output/reports/report_known_<YYYYMMDD>.json   — known employees only
        output/reports/report_unknown_<YYYYMMDD>.json — unknown persons only
        """
        import glob

        STREAM_GAP_THRESHOLD = 5 * 60  # 5 minutes = stream interruption

        # ── Partition 1: Known employees ──────────────────────────────────────
        known_sessions: dict = {}
        pattern_known = os.path.join(known_dir(), f"attendance_EMP*_{date_str}_*.json")
        for filepath in glob.glob(pattern_known):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                emp_id = data.get("employee_id")
                if not emp_id or emp_id == "Unknown" or emp_id is None:
                    continue
                emp_name = data.get("employee_name", "Unknown")
                entry = datetime.fromisoformat(data["entry_time"])
                exit_t = datetime.fromisoformat(data["exit_time"])
                work_sec = data.get("working_duration_seconds", 0.0)
                phone_sec = data.get("phone_use_duration_seconds", 0.0)
                phone_cnt = data.get("phone_use_count", 0)
                if emp_id not in known_sessions:
                    known_sessions[emp_id] = {"name": emp_name, "records": []}
                known_sessions[emp_id]["records"].append({
                    "entry": entry, "exit": exit_t,
                    "work_sec": work_sec, "phone_sec": phone_sec, "phone_cnt": phone_cnt,
                })
            except Exception as exc:
                logger.error("[AttendanceManager] Failed to process %s: %s", filepath, exc)

        known_report = []
        total_stream_interruptions = 0
        grand_total_work = 0.0
        grand_total_phone = 0.0

        for emp_id, info in known_sessions.items():
            records = sorted(info["records"], key=lambda r: r["entry"])
            total_work = 0.0
            total_phone = 0.0
            total_phone_cnt = 0
            stops = []
            stop_num = 1
            seg_work = 0.0
            seg_phone = 0.0
            seg_entry = None
            seg_exit = None
            prev_exit = None
            interruptions = 0

            for rec in records:
                if prev_exit is not None and (rec["entry"] - prev_exit).total_seconds() > STREAM_GAP_THRESHOLD:
                    # Stream interrupted — close the current stop
                    stops.append({
                        "stop_number": stop_num,
                        "entry_time": seg_entry.strftime("%H:%M:%S") if seg_entry else "",
                        "exit_time": seg_exit.strftime("%H:%M:%S") if seg_exit else "",
                        "working_hours": round(seg_work / 3600.0, 4),
                        "mobile_usage_hours": round(seg_phone / 3600.0, 4),
                    })
                    stop_num += 1
                    interruptions += 1
                    seg_work = 0.0
                    seg_phone = 0.0
                    seg_entry = rec["entry"]

                if seg_entry is None:
                    seg_entry = rec["entry"]

                seg_work += rec["work_sec"]
                seg_phone += rec["phone_sec"]
                seg_exit = rec["exit"]
                total_work += rec["work_sec"]
                total_phone += rec["phone_sec"]
                total_phone_cnt += rec["phone_cnt"]
                prev_exit = rec["exit"]

            # Append final stop
            if seg_work > 0 or seg_phone > 0 or seg_entry is not None:
                stops.append({
                    "stop_number": stop_num,
                    "entry_time": seg_entry.strftime("%H:%M:%S") if seg_entry else "",
                    "exit_time": seg_exit.strftime("%H:%M:%S") if seg_exit else "",
                    "working_hours": round(seg_work / 3600.0, 4),
                    "mobile_usage_hours": round(seg_phone / 3600.0, 4),
                })

            total_stream_interruptions += interruptions
            grand_total_work += total_work
            grand_total_phone += total_phone

            known_report.append({
                "employee_id": emp_id,
                "employee_name": info["name"],
                "total_working_hours": round(total_work / 3600.0, 4),
                "total_mobile_usage_hours": round(total_phone / 3600.0, 4),
                "total_phone_use_count": total_phone_cnt,
                "stream_stops": interruptions,
                "stops": stops,
            })

        # ── Partition 2: Unknown persons ──────────────────────────────────────
        unknown_report = []
        pattern_new = os.path.join(unknown_dir(), f"unknown_track_*_{date_str}_*.json")
        pattern_legacy = os.path.join(unknown_dir(), f"attendance_Unknown_*_{date_str}_*.json")
        all_unknown_files = glob.glob(pattern_new) + glob.glob(pattern_legacy)
        for filepath in all_unknown_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entry = datetime.fromisoformat(data["entry_time"])
                exit_t = datetime.fromisoformat(data["exit_time"])
                dur_sec = data.get("working_duration_seconds", 0.0)
                unknown_report.append({
                    "track_id": data.get("track_id"),
                    "camera_id": data.get("camera_id", ""),
                    "first_seen": entry.strftime("%H:%M:%S"),
                    "last_seen": exit_t.strftime("%H:%M:%S"),
                    "duration_minutes": round(dur_sec / 60.0, 2),
                    "mobile_usage_seconds": data.get("phone_use_duration_seconds", 0.0),
                    "phone_use_count": data.get("phone_use_count", 0),
                })
            except Exception as exc:
                logger.error("[AttendanceManager] Failed to process unknown record %s: %s", filepath, exc)

        unknown_report.sort(key=lambda u: u["first_seen"])

        # ── Partition 3: Overall summary ──────────────────────────────────────
        overall_summary = {
            "total_known_employees": len(known_report),
            "total_unknown_persons": len(unknown_report),
            "total_stream_interruptions": total_stream_interruptions,
            "total_working_hours_all_employees": round(grand_total_work / 3600.0, 4),
            "total_mobile_usage_hours_all_employees": round(grand_total_phone / 3600.0, 4),
        }

        # ── Write master combined report ──────────────────────────────────────
        final_report = {
            "date": date_str,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "known_employees": known_report,
            "unknown_persons": unknown_report,
            "overall_summary": overall_summary,
        }

        master_path = os.path.join(reports_dir(), f"daily_report_{date_str}.json")
        known_path = os.path.join(reports_dir(), f"report_known_{date_str}.json")
        unknown_path = os.path.join(reports_dir(), f"report_unknown_{date_str}.json")

        def _write(path, data):
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info("[AttendanceManager] Report written → %s", path)
            except Exception as exc:
                logger.error("[AttendanceManager] Failed to write %s: %s", path, exc)

        _write(master_path, final_report)
        _write(known_path, {"date": date_str, "known_employees": known_report, "overall_summary": overall_summary})
        _write(unknown_path, {"date": date_str, "unknown_persons": unknown_report, "total_unknown_persons": len(unknown_report)})


