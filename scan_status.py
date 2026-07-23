"""Tiny thread-safe holder for the background scanner's current activity.

The scanner runs on a BackgroundScheduler thread while the web server serves
requests on another; this lets the UI poll what the parser is doing right now.
"""
import threading

_lock = threading.Lock()
# revision bumps whenever a job row is stored/updated, so an open dashboard
# knows when there's new data worth reloading for.
_state = {"active": False, "message": "", "revision": 0}


def set_status(message: str) -> None:
    with _lock:
        _state["active"] = True
        _state["message"] = message


def clear_status() -> None:
    with _lock:
        _state["active"] = False
        _state["message"] = ""


def bump_revision() -> None:
    with _lock:
        _state["revision"] += 1


def get_status() -> dict:
    with _lock:
        return dict(_state)
