#!/usr/bin/env python3
"""
Daemon script that processes message files from other hostnames and forwards them
to the local memmachine API instance.
"""

from email import message
import os
from os import listdir
from os.path import isdir, join
import sys
import json
import socket
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:
    print("Error: 'watchdog' library is required. Install it with: python3 -m pip install watchdog")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Error: 'requests' library is required. Install it with: python3 -m pip install requests")
    sys.exit(1)


# Configuration
API_URL = "http://localhost:8080/api/v2/memories"
CHECK_INTERVAL = 10  # seconds


def log_message(message: str) -> None:
    """Log a message to the console with thread ID."""
    print(f"[{threading.get_ident()}] {message}")


def get_state_file_path() -> Path:
    """Get the path to the state file based on current hostname."""
    dropbox_dir = os.getenv("DROPBOX_DATA_DIR")
    if not dropbox_dir:
        raise ValueError("DROPBOX_DATA_DIR environment variable is not set")
    
    hostname = socket.gethostname()
    return Path(dropbox_dir) / f"{hostname}.state"


def load_state_file(state_file: Path) -> Dict[str, str]:
    """Load the state file containing hostname->timestamp mappings."""
    if not state_file.exists():
        log_message(f"State file {state_file} does not exist, creating empty state")
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text("{}")
        return {}
    
    try:
        with open(state_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        log_message(f"Error: Failed to parse state file {state_file}: {e}")
        return {}
    except Exception as e:
        log_message(f"Error: Failed to read state file {state_file}: {e}")
        return {}


def compare_timestamps(timestamp1: str, timestamp2: str) -> bool:
    """
    Compare two ISO 8601 timestamps.
    Returns True if timestamp1 > timestamp2.
    """
    try:
        # Parse ISO 8601 timestamps
        # Replace 'Z' with '+00:00' for UTC timezone
        ts1_clean = timestamp1.replace('Z', '+00:00')
        ts2_clean = timestamp2.replace('Z', '+00:00')
        
        dt1 = datetime.fromisoformat(ts1_clean)
        dt2 = datetime.fromisoformat(ts2_clean)
        return dt1 > dt2
    except (ValueError, AttributeError, TypeError) as e:
        # Fallback to string comparison (ISO 8601 is lexicographically sortable)
        # This works because ISO 8601 format is designed to be sortable as strings
        return timestamp1 > timestamp2


def diff_timestamps(timestamp1: str, timestamp2: str) -> float:
    """
    Diff two ISO 8601 timestamps.
    Returns the difference in seconds between timestamp1 and timestamp2.
    """
    try:
        dt1 = datetime.fromisoformat(timestamp1.replace('Z', '+00:00'))
        dt2 = datetime.fromisoformat(timestamp2.replace('Z', '+00:00'))
        return (dt1 - dt2).total_seconds()
    except (ValueError, AttributeError, TypeError) as e:
        log_message(f"Error: Failed to diff timestamps {timestamp1} and {timestamp2}: {e}")
        return 0


def extract_timestamp_from_filename(filepath: Path) -> str:
    """Extract timestamp from filename (e.g., '2025-12-29T19:50:59.952Z.msg' -> '2025-12-29T19:50:59.952Z')."""
    return filepath.stem  # Gets filename without extension


def forward_message(msg_file: Path, api_url: str) -> bool:
    """Forward a message file to the API."""
    log_message(f"Forwarding message file: {msg_file}")
    
    if not msg_file.exists() or not msg_file.is_file():
        log_message(f"Error: Message file {msg_file} does not exist or is not a file")
        return False
    
    try:
        # Read the message file
        with open(msg_file, 'r') as f:
            message_data = json.load(f)
        
        # POST to API
        response = requests.post(api_url, json=message_data, timeout=30)

        current_timestamp = datetime.now(timezone.utc).isoformat()

        message_content = message_data["messages"][0]["content"]
        message_content_short = message_content[:min(len(message_content), 8)]
        if len(message_content) > 8:
            message_content_short += "..."
        
        if response.status_code >= 200 and response.status_code < 300:
            log_message(f"Forwarded {msg_file} (HTTP {response.status_code}) at timestamp {current_timestamp} (diff {diff_timestamps(current_timestamp, msg_file.stem)})")
            ## On mac, shows a system tray notification when message is synced ##
            if sys.platform == "darwin":
                subprocess.run([
                    "osascript",
                    "-e",
                    f'display notification "Added message: {message_content_short}" with title "MemMachine Sync"'
                ])
            return True
        else:
            log_message(f"Failed to forward message from {msg_file} (HTTP {response.status_code})")
            try:
                error_response = response.json()
                log_message(f"API response: {error_response}")
            except:
                log_message(f"API response: {response.text}")
            return False
    
    except json.JSONDecodeError as e:
        log_message(f"Error: Failed to parse message file {msg_file} as JSON: {e}")
        return False
    except requests.exceptions.RequestException as e:
        log_message(f"Error: Failed to send request to API: {e}")
        return False
    except Exception as e:
        log_message(f"Error: Unexpected error while forwarding message: {e}")
        return False


state = {}


def dump_to_state_file(state: dict):
    try:
        with open(get_state_file_path(), 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Updated state file: {get_state_file_path()}")
    except Exception as e:
        print(f"Error: Failed to save state file: {e}", file=sys.stderr)


class MessageEventHandler(FileSystemEventHandler):
    def on_created(self, event: FileSystemEvent) -> None:
        message_path = Path(event.src_path)

        if event.is_directory:
            return

        if message_path.suffix == ".msg":
            ts = message_path.stem
            host = message_path.parent.name
            
            if not forward_message(event.src_path, API_URL):
                log_message(f"Error: Failed to forward message from {event.src_path}")
                return

            if compare_timestamps(ts, state[host]):
                state[host] = ts

        return


def add_new_messages():
    saved_state = load_state_file(get_state_file_path())
    ## Ignore the current hostname + memmachine directory (allows me to keep this project in dropbox, not have to mirror changes to the repo) ##
    ignore_list = [socket.gethostname(), "memmachine", "mmemmachine"]
    dropbox_dir = os.getenv("DROPBOX_DATA_DIR")
    if not dropbox_dir:
        log_message("Error: DROPBOX_DATA_DIR environment variable is not set")
        return

    dropbox_dir_path = Path(dropbox_dir)

    for f in dropbox_dir_path.iterdir():
        if not f.is_dir():
            continue

        if f.name in ignore_list:
            log_message(f"Ignoring directory {f.name}")
            continue

        host = str(f.name)
        msg_files = list[Any](f.glob("*.msg"))
        last_saved_timestamp = saved_state.get(host)
        if last_saved_timestamp is None:
            last_saved_timestamp = "1970-01-01T00:00:00.000Z"
        most_recent_timestamp = last_saved_timestamp

        log_message(f"Processing messages from {host}")

        for msg_file in msg_files:
            file_timestamp = extract_timestamp_from_filename(msg_file)
            
            # Compare timestamps: process if file_timestamp > last_saved_timestamp
            if compare_timestamps(file_timestamp, last_saved_timestamp):
                if not forward_message(msg_file, API_URL):
                    log_message(f"Error: Failed to forward message from {msg_file}")
                    return

            ## Update most_recent_timestamp if file_timestamp is greater
            ## Done seperately from previous timestamp in case the msg_files are not processed in order
            if compare_timestamps(file_timestamp, most_recent_timestamp):
                most_recent_timestamp = file_timestamp
        
        state[host] = most_recent_timestamp
    
    return


def watch_for_messages() -> None:
    """ Propagate all messages that are in the state folders. """
    dropbox_dir = os.getenv("DROPBOX_DATA_DIR")
    if not dropbox_dir:
        log_message("Error: DROPBOX_DATA_DIR environment variable is not set")
        return

    observer = Observer()
    message_event_handler = MessageEventHandler()
    observer.schedule(message_event_handler, str(dropbox_dir), recursive=True)
    observer.start()
    try:
        while True:
            observer.join(1)
    except KeyboardInterrupt:
        observer.stop()
        dump_to_state_file(state)
    observer.join()


if __name__ == "__main__":
    try:
        add_new_messages()
        watch_for_messages()
    except KeyboardInterrupt:
        log_message("\nShutting down...")
        sys.exit(0)
    except Exception as e:
        log_message(f"Error in main loop: {e}")
