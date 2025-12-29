#!/usr/bin/env python3
"""
Daemon script that processes message files from other hostnames and forwards them
to the local memmachine API instance.
"""

import os
import sys
import json
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

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


def process_hostname_messages(
    remote_hostname: str,
    last_timestamp: str,
    dropbox_dir: Path,
    api_url: str
) -> None:
    """Process all message files for a specific hostname."""
    host_dir = dropbox_dir / remote_hostname
    
    if not host_dir.exists() or not host_dir.is_dir():
        print(f"Directory {host_dir} does not exist, skipping hostname {remote_hostname}")
        return
    
    print(f"Processing messages for hostname: {remote_hostname} (last timestamp: {last_timestamp})")
    
    # Find all .msg files in the hostname directory
    msg_files = list(host_dir.glob("*.msg"))
    
    for msg_file in msg_files:
        file_timestamp = extract_timestamp_from_filename(msg_file)
        
        # Compare timestamps: process if file_timestamp > last_timestamp
        if compare_timestamps(file_timestamp, last_timestamp):
            forward_message(msg_file, api_url)


def main() -> None:
    """Main function to process messages."""
    print("Starting memmachine compose daemon")
    
    # Check if DROPBOX_DATA_DIR is set
    dropbox_dir = os.getenv("DROPBOX_DATA_DIR")
    if not dropbox_dir:
        print("Error: DROPBOX_DATA_DIR environment variable is not set", file=sys.stderr)
        return
    
    dropbox_path = Path(dropbox_dir)
    current_hostname = socket.gethostname()
    state_file = get_state_file_path()
    
    print(f"State file: {state_file}")
    print(f"Current hostname: {current_hostname}")
    
    # Load state file
    state = load_state_file(state_file)
    
    if not state:
        print("No hostnames found in state file")
        return
    
    # Process each hostname (excluding current hostname)
    for hostname, timestamp in state.items():
        if hostname == current_hostname:
            print(f"Skipping current hostname: {hostname}")
            continue
        
        if not timestamp or timestamp == "null":
            print(f"No timestamp found for hostname {hostname}, skipping")
            continue
        
        process_hostname_messages(hostname, timestamp, dropbox_path, API_URL)


if __name__ == "__main__":
    # Run the main function in a loop
    while True:
        try:
            main()
        except KeyboardInterrupt:
            print("\nShutting down...")
            sys.exit(0)
        except Exception as e:
            print(f"Error in main loop: {e}", file=sys.stderr)
        
        time.sleep(CHECK_INTERVAL)

