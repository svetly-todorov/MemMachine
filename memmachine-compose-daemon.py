#!/usr/bin/env python3
"""
Daemon script that processes message files from other hostnames and forwards them
to the local memmachine API instance.
"""

import os
from os import listdir
from os.path import isdir, join
import sys
import json
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    print("Error: 'requests' library is required. Install it with: pip install requests")
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
        message_content_short_length = 16
        message_content_short = message_content[:min(len(message_content), message_content_short_length)]
        if len(message_content) > message_content_short_length:
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


# def update_state_file_with_new_hostnames(
#     state_file: Path,
#     dropbox_dir: Path,
#     current_hostname: str
# ) -> Dict[str, str]:
#     """Update state file with any new hostname directories found."""
#     state = load_state_file(state_file)
    
#     # Discover all hostname directories
#     discovered_hostnames = discover_hostname_directories(dropbox_dir, current_hostname)
    
#     updated = False
#     for hostname in discovered_hostnames:
#         # Skip current hostname
#         if hostname == current_hostname:
#             continue
        
#         # If hostname is not in state, add it
#         if hostname not in state:
#             # Use an early timestamp (epoch) so we'll process future messages
#             state[hostname] = "1970-01-01T00:00:00.000Z"
#             log_message(f"Added new hostname '{hostname}' to state file with default timestamp (no .msg files found)")
#             updated = True
    
#     # Save updated state file if changes were made
#     if updated:
#         return dump_to_state_file(state, state_file)
    
#     return state


def list_folder(path: Path) -> None:
    """Call list_folder API."""
    strpath = str(path)
    url = 'https://api.dropboxapi.com/2/files/list_folder'
    data = json.dumps({
        "include_deleted": False,
        "include_has_explicit_shared_members": False,
        "include_media_info": False,
        "include_mounted_folders": True,
        "include_non_downloadable_files": True,
        "path": f"/{strpath}",
        "recursive": False
    })
    response = requests.post(url, data=data, headers={'Authorization': f'Bearer {os.getenv("DROPBOX_ACCESS_TOKEN")}', 'Content-Type': 'application/json'})
    if response.status_code != 200:
        log_message(f"Error: Did not get status 200 for list_folder at /{strpath}, instead {response.status_code} with json: {response.json()}")
        exit(1)
    log_message("---")
    log_message(f"List folder response for {strpath}:")
    log_message(response.json())
    log_message("---")


def get_latest_cursor(path: Path) -> str:
    """Get the latest cursor for the path. Path is relative to the root of the dropbox directory."""
    strpath = str(path)
    url = 'https://api.dropboxapi.com/2/files/list_folder/get_latest_cursor'
    data = json.dumps({
        "include_deleted": False,
        "include_has_explicit_shared_members": False,
        "include_media_info": False,
        "include_mounted_folders": True,
        "include_non_downloadable_files": True,
        "path": f"/{strpath}",
        "recursive": False
    })
    response = requests.post(url, data=data, headers={'Authorization': f'Bearer {os.getenv("DROPBOX_ACCESS_TOKEN")}', 'Content-Type': 'application/json'})
    if response.status_code != 200:
        log_message(f"Error: Did not get status 200 for get_latest_cursor at /{strpath}, instead {response.status_code} with json: {response.json()}")
        exit(1)
    ## log_message(f"get_latest_cursor response for {strpath}: {response.json()}")
    if 'cursor' not in response.json():
        log_message(f"Error: Did not get cursor for get_latest_cursor at /{strpath}: {response.json()}")
        exit(1)
    ## log_message(f"Found cursor: {response.json()['cursor']} for {strpath}")
    return response.json()['cursor']


def get_path_in_dropbox(path: Path) -> Path:
    """Get the path relative to the root of the dropbox directory."""
    dropbox_data_dir = os.getenv("DROPBOX_DATA_DIR")
    if not dropbox_data_dir:
        log_message("Error: DROPBOX_DATA_DIR environment variable is not set")
        return
    
    return path.relative_to(Path(dropbox_data_dir))


def longpoll(path: Path) -> None:
    """Long poll for new subdirectories under path. Path should be relative to the root of the dropbox directory."""

    longpoll_timeout = 480
    reset = True
    cursor = None

    while True:
        if reset:
            cursor = get_latest_cursor(path)
            if cursor is None:
                log_message(f"Error: Failed to get latest cursor for {path}")
                return
        
        log_message(f"Long polling at {path} with cursor {cursor} and max timeout {longpoll_timeout}s")

        url = 'https://notify.dropboxapi.com/2/files/list_folder/longpoll'
        data = json.dumps({
            "cursor": cursor,
            "timeout": longpoll_timeout
        })
        response = requests.post(url, data=data, headers={'Content-Type': 'application/json'})
        if response.status_code == 200:
            response_json = response.json()
            if 'changes' in response_json:
                if response_json['changes'] == True:
                    log_message(f"Changes found at {path}, exiting longpoll...")
                    return
                else:
                    log_message(f"No changes found at {path}, continuing to wait...")
                    continue
            elif 'reset' in response_json:
                log_message(f"Cursor {cursor} is expired and must reset")
                reset = True
                continue
            else:
                log_message(f"Error: Did not recognize condition after 200 success: got {response_json}")
                exit(1)
        else:
            log_message(f"Error: Failed to long poll at {path}, status code {response.status_code}")
            exit(1)


def process_new_messages(
    remote_hostname: str,
    previous_timestamp: str,
    dropbox_dir: Path,
    api_url: str
) -> Optional[str]:
    """Process all message files for a specific hostname."""
    host_dir = dropbox_dir / remote_hostname
    most_recent_timestamp = previous_timestamp

    if not host_dir.exists() or not host_dir.is_dir():
        log_message(f"Directory {host_dir} does not exist on the local host, therefore continuing to wait...")
        return

    log_message(f"Processing messages for hostname: {remote_hostname} (since previous timestamp: {previous_timestamp})")

    # Find all .msg files in the given remote-hostname directory
    msg_files = list(host_dir.glob("*.msg"))

    for msg_file in msg_files:
        file_timestamp = extract_timestamp_from_filename(msg_file)
        
        # Compare timestamps: process if file_timestamp > previous_timestamp
        if compare_timestamps(file_timestamp, previous_timestamp):
            if not forward_message(msg_file, api_url):
                log_message(f"Error: Failed to forward message from {msg_file}")
                return

        ## Update most_recent_timestamp if file_timestamp is greater
        ## Done seperately from previous timestamp in case the msg_files are not processed in order
        if compare_timestamps(file_timestamp, most_recent_timestamp):
            most_recent_timestamp = file_timestamp

    return most_recent_timestamp


def process_messages_thread(
    remote_hostname: str,
    dropbox_dir: Path,
    api_url: str
) -> None:
    """Process messages for a specific hostname."""

    # Get the most recent timestamp from dropbox_dir/hostname.state, or use 1970-01-01T00:00:00.000Z if it doesn't exist
    state_file = get_state_file_path()
    state = load_state_file(state_file)
    log_message(f"State file: {state_file}")

    previous_timestamp = "1970-01-01T00:00:00.000Z"

    if remote_hostname not in state:
        ## In the future, we have to lock the state dictionary + json file in order to dump ##
        state[remote_hostname] = "1970-01-01T00:00:00.000Z"
        previous_timestamp = "1970-01-01T00:00:00.000Z"
    else:
        previous_timestamp = state[remote_hostname]

    while True:
        most_recent_timestamp = process_new_messages(remote_hostname, previous_timestamp, dropbox_dir, api_url)
        if most_recent_timestamp is not None:
            state[remote_hostname] = most_recent_timestamp
            previous_timestamp = most_recent_timestamp
            log_message(f"Updated {remote_hostname} state with more recent timestamp: {previous_timestamp}")
            ## TODO: Dump the newest most recent timestamp to the state file ##
        longpoll(path=get_path_in_dropbox(dropbox_dir / remote_hostname))


def watch_for_new_hosts() -> None:
    """Loop over directoies under DROPBOX_DATA_DIR; use /longpoll to check if there are new subdirectories; start a thread for each"""
    dropbox_dir = os.getenv("DROPBOX_DATA_DIR")
    if not dropbox_dir:
        log_message("Error: DROPBOX_DATA_DIR environment variable is not set")
        return
    
    dropbox_path = Path(dropbox_dir)
    state_file = get_state_file_path()

    monitored_directories = []

    ## Ignore the current hostname + memmachine directory (allows me to keep this project in dropbox, not have to mirror changes to the repo) ##
    ignore_list = [socket.gethostname(), "memmachine", "mmemmachine"]

    ## For debug, list the contents of dropbox root dir ##
    list_folder(Path("/"))

    while True:
        for f in dropbox_path.iterdir():
            if not f.is_dir():
                continue

            log_message(f"Found directory: {f}")

            if f.name not in monitored_directories and f.name not in ignore_list:
                log_message(f"Starting a thread to monitor directory {f}")
                remote_hostname = str(f.name)
                monitored_directories.append(f.name)
                thread = threading.Thread(target=process_messages_thread, args=(remote_hostname, dropbox_path, API_URL))
                thread.start()

        ## Watch the data directory for new hosts ##
        longpoll(path=(Path("/")))


if __name__ == "__main__":
    try:
        watch_for_new_hosts()
    except KeyboardInterrupt:
        log_message("\nShutting down...")
        sys.exit(0)
    except Exception as e:
        log_message(f"Error in main loop: {e}")

