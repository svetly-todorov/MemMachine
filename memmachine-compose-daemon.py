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
import threading
import time
from datetime import datetime
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
        print(f"State file {state_file} does not exist, creating empty state")
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text("{}")
        return {}
    
    try:
        with open(state_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse state file {state_file}: {e}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"Error: Failed to read state file {state_file}: {e}", file=sys.stderr)
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


def extract_timestamp_from_filename(filepath: Path) -> str:
    """Extract timestamp from filename (e.g., '2025-12-29T19:50:59.952Z.msg' -> '2025-12-29T19:50:59.952Z')."""
    return filepath.stem  # Gets filename without extension


def forward_message(msg_file: Path, api_url: str) -> bool:
    """Forward a message file to the API."""
    print(f"Forwarding message file: {msg_file}")
    
    if not msg_file.exists() or not msg_file.is_file():
        print(f"Error: Message file {msg_file} does not exist or is not a file", file=sys.stderr)
        return False
    
    try:
        # Read the message file
        with open(msg_file, 'r') as f:
            message_data = json.load(f)
        
        # POST to API
        response = requests.post(api_url, json=message_data, timeout=30)
        
        if response.status_code >= 200 and response.status_code < 300:
            print(f"Successfully forwarded message from {msg_file} (HTTP {response.status_code})")
            return True
        else:
            print(f"Failed to forward message from {msg_file} (HTTP {response.status_code})", file=sys.stderr)
            try:
                error_response = response.json()
                print(f"API response: {error_response}", file=sys.stderr)
            except:
                print(f"API response: {response.text}", file=sys.stderr)
            return False
    
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse message file {msg_file} as JSON: {e}", file=sys.stderr)
        return False
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to send request to API: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error: Unexpected error while forwarding message: {e}", file=sys.stderr)
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
#             print(f"Added new hostname '{hostname}' to state file with default timestamp (no .msg files found)")
#             updated = True
    
#     # Save updated state file if changes were made
#     if updated:
#         return dump_to_state_file(state, state_file)
    
#     return state


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
        print(f"Error: Failed to get latest cursor for /{strpath}: {response.status_code} with error: {response.text}", file=sys.stderr)
        return None
    response_json = response.json()
    if 'cursor' not in response_json:
        print(f"Error: Failed to get latest cursor for /{strpath}: {response_json}", file=sys.stderr)
        return None
    return response_json['cursor']


def longpoll(path: Path) -> None:
    """Long poll for new subdirectories under path."""

    longpoll_timeout = 30
    reset = True
    cursor = None

    print(f"Long polling for changes at /{str(path)}")

    while True:
        if reset:
            cursor = get_latest_cursor(path)
            if cursor is None:
                print(f"Error: Failed to get latest cursor for {path}", file=sys.stderr)
                return

        url = 'https://api.dropboxapi.com/2/files/list_folder/longpoll'
        data = json.dumps({
            "cursor": cursor,
            "timeout": longpoll_timeout
        })
        response = requests.post(url, data=data, headers={'Content-Type': 'application/json'})
        if response.status_code == 200:
            response_json = response.json()
            if 'changes' in response_json:
                if response_json['changes'] == True:
                    print(f"Changes found at {path}, exiting longpoll...")
                    return
                else:
                    print(f"No changes found at {path}, continuing to wait...")
                    continue
            elif 'reset' in response_json:
                print(f"Cursor {cursor} is expired and must reset")
                reset = True
                continue
            else:
                print(f"Error: Failed to long poll at {path}: {response_json}", file=sys.stderr)
                return


def process_new_messages(
    remote_hostname: str,
    last_timestamp: str,
    dropbox_dir: Path,
    api_url: str
) -> Optional[str]:
    """Process all message files for a specific hostname."""
    host_dir = dropbox_dir / remote_hostname
    
    if not host_dir.exists() or not host_dir.is_dir():
        print(f"Directory {host_dir} does not exist on the local host, therefore continuing to wait...")
        return
    
    print(f"Processing messages for hostname: {remote_hostname} (last timestamp: {last_timestamp})")
    
    while True:
        # Find all .msg files in the hostname directory
        msg_files = list(host_dir.glob("*.msg"))

        most_recent_timestamp = None
        
        for msg_file in msg_files:
            file_timestamp = extract_timestamp_from_filename(msg_file)
            
            # Compare timestamps: process if file_timestamp > last_timestamp
            if compare_timestamps(file_timestamp, last_timestamp):
                forward_message(msg_file, api_url)
                most_recent_timestamp = file_timestamp


def process_messages_thread(
    remote_hostname: str,
    dropbox_dir: Path,
    api_url: str
) -> None:
    """Process messages for a specific hostname."""

    # Get the most recent timestamp from dropbox_dir/hostname.state, or use 1970-01-01T00:00:00.000Z if it doesn't exist
    state_file = get_state_file_path()
    state = load_state_file(state_file)
    print(f"State file: {state_file}")

    remote_hostname_path = Path(f"{dropbox_dir.parts[-1]}/{remote_hostname}")

    last_timestamp = "1970-01-01T00:00:00.000Z"

    if remote_hostname not in state:
        ## In the future, we have to lock the state dictionary + json file to dump ##
        state[remote_hostname] = "1970-01-01T00:00:00.000Z"
        last_timestamp = "1970-01-01T00:00:00.000Z"
    else:
        last_timestamp = state[remote_hostname]

    while True:
        process_new_messages(remote_hostname, last_timestamp, dropbox_dir, api_url)
        ## A fast way of getting the last two path elements: https://stackoverflow.com/a/69466662 ##
        longpoll(path=remote_hostname_path)


def watch_for_new_hosts() -> None:
    """Loop over directoies under DROPBOX_DATA_DIR; use /longpoll to check if there are new subdirectories; start a thread for each"""
    dropbox_dir = os.getenv("DROPBOX_DATA_DIR")
    if not dropbox_dir:
        print("Error: DROPBOX_DATA_DIR environment variable is not set", file=sys.stderr)
        return
    
    dropbox_path = Path(dropbox_dir)
    current_hostname = socket.gethostname()
    state_file = get_state_file_path()

    monitored_directories = []

    while True:
        for f in dropbox_path.iterdir():
            if not f.is_dir():
                continue

            if f not in monitored_directories and f != current_hostname:
                monitored_directories.append(f)
                # Spin up a thread to process f
                thread = threading.Thread(target=process_messages_thread, args=(f, state_file, dropbox_path, API_URL))
                thread.start()

        longpoll(path=(dropbox_path).parts[-1])


if __name__ == "__main__":
    try:
        watch_for_new_hosts()
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"Error in main loop: {e}", file=sys.stderr)

