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


def get_earliest_timestamp_from_hostname_dir(host_dir: Path) -> Optional[str]:
    """Get the earliest timestamp from .msg files in a hostname directory."""
    msg_files = list(host_dir.glob("*.msg"))
    if not msg_files:
        return None
    
    earliest_timestamp = None
    for msg_file in msg_files:
        file_timestamp = extract_timestamp_from_filename(msg_file)
        # If earliest is None, or if file_timestamp is earlier than earliest (i.e., earliest > file)
        if earliest_timestamp is None or compare_timestamps(earliest_timestamp, file_timestamp):
            earliest_timestamp = file_timestamp
    
    return earliest_timestamp


def discover_hostname_directories(dropbox_dir: Path, current_hostname: str) -> List[str]:
    """Discover hostname directories in the dropbox directory."""
    hostnames = []
    
    # Directories to exclude (non-hostname directories)
    exclude_dirs = {
        "docker_volumes_lockfolder",
        "docker_volumes",
    }
    
    if not dropbox_dir.exists() or not dropbox_dir.is_dir():
        return hostnames
    
    for item in dropbox_dir.iterdir():
        # Skip if not a directory
        if not item.is_dir():
            continue
        
        # Skip excluded directories
        if item.name in exclude_dirs:
            continue
        
        # Skip if it's a state file (shouldn't be a directory, but just in case)
        if item.name.endswith('.state'):
            continue
        
        # Consider it a hostname directory
        hostnames.append(item.name)
    
    return hostnames


def dump_to_state_file(state: Dict[str, str], state_file: Path) -> bool:
    """Dump the state to the state file if changes were made."""
    try:
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Error: Failed to save state file: {e}", file=sys.stderr)
        return False
    return True


def update_state_file_with_new_hostnames(
    state_file: Path,
    dropbox_dir: Path,
    current_hostname: str
) -> Dict[str, str]:
    """Update state file with any new hostname directories found."""
    state = load_state_file(state_file)
    
    # Discover all hostname directories
    discovered_hostnames = discover_hostname_directories(dropbox_dir, current_hostname)
    
    updated = False
    for hostname in discovered_hostnames:
        # Skip current hostname
        if hostname == current_hostname:
            continue
        
        # If hostname is not in state, add it
        if hostname not in state:
            # Use an early timestamp (epoch) so we'll process future messages
            state[hostname] = "1970-01-01T00:00:00.000Z"
            print(f"Added new hostname '{hostname}' to state file with default timestamp (no .msg files found)")
            updated = True
    
    # Save updated state file if changes were made
    if updated:
        return dump_to_state_file(state, state_file)
    
    return state


def process_hostname_messages(
    remote_hostname: str,
    last_timestamp: str,
    dropbox_dir: Path,
    api_url: str
) -> Optional[str]:
    """Process all message files for a specific hostname."""
    host_dir = dropbox_dir / remote_hostname
    
    if not host_dir.exists() or not host_dir.is_dir():
        print(f"Directory {host_dir} does not exist, skipping hostname {remote_hostname}")
        return
    
    print(f"Processing messages for hostname: {remote_hostname} (last timestamp: {last_timestamp})")
    
    # Find all .msg files in the hostname directory
    msg_files = list(host_dir.glob("*.msg"))

    most_recent_timestamp = None
    
    for msg_file in msg_files:
        file_timestamp = extract_timestamp_from_filename(msg_file)
        
        # Compare timestamps: process if file_timestamp > last_timestamp
        if compare_timestamps(file_timestamp, last_timestamp):
            forward_message(msg_file, api_url)
            most_recent_timestamp = file_timestamp

    return most_recent_timestamp


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
    
    # Update state file with any new hostname directories discovered
    state = update_state_file_with_new_hostnames(state_file, dropbox_path, current_hostname)
    updated = False

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
        
        most_recent_timestamp = process_hostname_messages(hostname, timestamp, dropbox_path, API_URL)
        if most_recent_timestamp:
            state[hostname] = most_recent_timestamp
            updated = True

    # Save updated state if there there were more recent messages in the subdirectories
    if updated:
        dump_to_state_file(state, state_file)


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

