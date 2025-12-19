#!/usr/bin/env python3
"""
Script to modify docker-compose.yml with Dropbox-aware volume path handling.
Attempts to acquire lock by creating a folder via Dropbox API (folder creation is atomic).
Falls back to local copy if lock acquisition fails (folder already exists).
"""

import os
import sys
import re
import shutil
import subprocess
import socket
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple

try:
    import requests
except ImportError:
    print("Error: 'requests' library is required. Install it with: pip install requests")
    sys.exit(1)


# Configuration
INPUT_FILE = "docker-compose.yml"
# The current directory name should be here
LOCKFOLDER_NAME = "docker_volumes_lockfolder"
LOCKFOLDER_PATH = f"{os.getenv('DROPBOX_DATA_DIR')}/{LOCKFOLDER_NAME}"
# Dropbox path for the lockfolder is relative to the root of my personal directory
DROPBOX_LOCKFOLDER_PATH = f"/{LOCKFOLDER_NAME}"
VOLUMES_DIR = f"{os.getenv('DROPBOX_DATA_DIR')}/docker_volumes"
VOLUMES_DIR_LOCAL = "./docker_volumes_local"

# Volume subdirectories to create
VOLUME_SUBDIRS = [
    "postgres_data",
    "neo4j_data",
    "neo4j_logs",
    "neo4j_import",
    "neo4j_plugins",
    "memmachine_logs",
]


def get_user_ids() -> Tuple[int, int]:
    """Get current user UID and GID."""
    return os.getuid(), os.getgid()


def check_dropbox_data_directory() -> bool:
    """
    Check if the Dropbox data directory is set.
    """
    return os.environ.get("DROPBOX_DATA_DIR") is not None


def check_dropbox_api_token() -> bool:
    """
    Check if the Dropbox API token is set.
    """
    return os.environ.get("DROPBOX_ACCESS_TOKEN") is not None


def create_dropbox_lockfile() -> None:
    """
    Create a file inside of the lockfolder.
    """
    with open(f"{LOCKFOLDER_PATH}/lockfile.txt", "w+") as file:
        file.write(f"{socket.gethostname()}")


def check_dropbox_lockfile() -> Tuple[bool, bool]:
    """
    Check if the lockfile exists and is owned by the current host.
    """
    lockfile_path = f"{LOCKFOLDER_PATH}/lockfile.txt"
    if not os.path.exists(lockfile_path):
        print(f"Error: lockfile not found: {lockfile_path}")
        return False, False
    try:
        with open(lockfile_path, "r") as file:
            hostname = file.read()
            print(f"Lockfile {lockfile_path} is owned by {hostname}")
            return True, hostname == socket.gethostname()
    except Exception as e:
        print(f"Error checking lockfile: {e}")
        return False, False


def create_dropbox_lockfolder() -> Tuple[bool, Optional[str]]:
    """
    Attempt to acquire lock by creating a folder using Dropbox API.
    Folder creation is atomic - if it already exists, the operation fails,
    which means the lock is held by another process/machine.
    
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    access_token = os.environ.get("DROPBOX_ACCESS_TOKEN")
    if not access_token:
        return False, "DROPBOX_ACCESS_TOKEN environment variable not set"
    
    try:
        # Prepare Dropbox API request to create folder
        url = "https://api.dropboxapi.com/2/files/create_folder_v2"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "path": DROPBOX_LOCKFOLDER_PATH,
            "autorename": False
        }
        
        # Make API call
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            # Successfully created folder = successfully acquired lock
            result = response.json()
            print(f"Successfully acquired Dropbox lock (created folder: {result.get('metadata', {}).get('name', LOCKFOLDER_NAME)})")
            return True, None
        else:
            # API call failed - likely folder already exists (lock held)
            error_msg = response.text
            try:
                error_json = response.json()
                error_summary = error_json.get("error_summary", "")
                # Check if it's a path/conflict error (folder already exists)
                if "path/conflict" in error_summary or "folder_already_exists" in error_summary.lower():
                    return False, f"Lock folder already exists - lock is held by another process/machine"
                error_msg = error_summary
            except (ValueError, KeyError):
                pass
            return False, f"Dropbox API error (status {response.status_code}): {error_msg}"
            
    except requests.exceptions.RequestException as e:
        return False, f"Network error connecting to Dropbox API: {e}"
    except Exception as e:
        return False, f"Unexpected error during lock acquisition: {e}"


def wait_for_dropbox_lockfolder() -> bool:
    """
    Wait for the lockfolder to be created.
    """
    while not os.path.exists(LOCKFOLDER_PATH):
        print(f"Waiting for lockfolder to appear locally at {LOCKFOLDER_PATH}...")
        time.sleep(1)
    return True


def attempt_dropbox_lock() -> Tuple[bool, Optional[str]]:
    """
    Attempt to acquire lock by creating a folder using Dropbox API.
    """
    lock_acquired, lock_error = create_dropbox_lockfolder()
    if lock_acquired:
        wait_for_dropbox_lockfolder()
        create_dropbox_lockfile()
        return True, None
    else:
        return False, lock_error


def release_dropbox_lock() -> bool:
    """
    Delete the lockfolder via Dropbox API to release the lock.
    
    Returns:
        True if successful, False otherwise
    """
    access_token = os.environ.get("DROPBOX_ACCESS_TOKEN")
    if not access_token:
        print("Error: DROPBOX_ACCESS_TOKEN environment variable not set", file=sys.stderr)
        return False
    
    try:
        # Prepare Dropbox API request to delete folder
        url = "https://api.dropboxapi.com/2/files/delete_v2"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "path": DROPBOX_LOCKFOLDER_PATH
        }
        
        # Make API call
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            print(f"Successfully deleted lockfolder: {LOCKFOLDER_NAME}")
            return True
        else:
            # API call failed
            error_msg = response.text
            try:
                error_json = response.json()
                error_summary = error_json.get("error_summary", "")
                # Check if folder doesn't exist (already deleted or never existed)
                if "path_lookup/not_found" in error_summary or "not_found" in error_summary.lower():
                    print(f"Lockfolder does not exist (may have already been deleted): {LOCKFOLDER_NAME}")
                    return True  # Consider this success - folder is gone
                error_msg = error_summary
            except (ValueError, KeyError):
                pass
            print(f"Error deleting lockfolder (status {response.status_code}): {error_msg}", file=sys.stderr)
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Network error connecting to Dropbox API: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error during lock teardown: {e}", file=sys.stderr)
        return False


def copy_volumes_to_local() -> bool:
    """
    Copy docker_volumes directory to docker_volumes_local.
    
    Returns:
        True if successful, False otherwise
    """
    volumes_path = Path(VOLUMES_DIR)
    local_path = Path(VOLUMES_DIR_LOCAL)
    
    try:
        if volumes_path.exists() and volumes_path.is_dir():
            print(f"Copying {VOLUMES_DIR} to {VOLUMES_DIR_LOCAL}")
            if local_path.exists():
                shutil.rmtree(local_path)
            shutil.copytree(volumes_path, local_path)
            return True
        else:
            print(f"Creating {VOLUMES_DIR_LOCAL} directory structure")
            local_path.mkdir(parents=True, exist_ok=True)
            return True
    except Exception as e:
        print(f"Error copying volumes directory: {e}", file=sys.stderr)
        return False


def ensure_volume_directories(volumes_path: str) -> bool:
    """
    Create volume subdirectories if they don't exist.
    
    Args:
        volumes_path: Base path for volumes
        
    Returns:
        True if successful, False otherwise
    """
    try:
        base_path = Path(volumes_path)
        for subdir in VOLUME_SUBDIRS:
            (base_path / subdir).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating volume directories: {e}", file=sys.stderr)
        return False


def modify_docker_compose(input_file: str, volumes_path: str, uid: int, gid: int) -> bool:
    """
    Modify docker-compose.yml file with volume path replacements and user settings.
    
    Args:
        input_file: Path to docker-compose.yml
        volumes_path: Path to use for volumes (docker_volumes or docker_volumes_local)
        uid: User ID to set
        gid: Group ID to set
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the file
        with open(input_file, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Replace volume paths
        volume_replacements = {
            'postgres_data': f'{volumes_path}/postgres_data',
            'neo4j_data': f'{volumes_path}/neo4j_data',
            'neo4j_logs': f'{volumes_path}/neo4j_logs',
            'neo4j_import': f'{volumes_path}/neo4j_import',
            'neo4j_plugins': f'{volumes_path}/neo4j_plugins',
            'memmachine_logs': f'{volumes_path}/memmachine_logs',
        }
        
        for volume_name, volume_path in volume_replacements.items():
            # Pattern: lines starting with spaces, dash, space, then volume_name:
            pattern = rf'^(\s+-\s+)([A-Za-z0-9\._\/]*{re.escape(volume_name)}):'
            replacement = rf'\1{volume_path}:'
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        # Remove user: lines for postgres and neo4j services
        # Match from "  postgres:" or "  neo4j:" to next service (line starting with "  [a-z]")
        lines = content.split('\n')
        in_postgres = False
        in_neo4j = False
        new_lines = []
        
        for i, line in enumerate(lines):
            # Check if we're entering a service block
            if re.match(r'^  postgres:', line):
                in_postgres = True
                in_neo4j = False
                new_lines.append(line)
            elif re.match(r'^  neo4j:', line):
                in_neo4j = True
                in_postgres = False
                new_lines.append(line)
            # Check if we're leaving the service block (next service starts)
            elif re.match(r'^  [a-z]', line) and (in_postgres or in_neo4j):
                in_postgres = False
                in_neo4j = False
                new_lines.append(line)
            # Skip user: lines within postgres or neo4j blocks
            elif (in_postgres or in_neo4j) and re.match(r'^    user:', line):
                continue  # Skip this line
            else:
                new_lines.append(line)
        
        content = '\n'.join(new_lines)
        
        # Add user: line after container_name for postgres and neo4j
        user_line = f'    user: "{uid}:{gid}"'
        for container in ['memmachine-postgres', 'memmachine-neo4j']:
            pattern = rf'(container_name: {re.escape(container)}\n)'
            replacement = rf'\1{user_line}\n'
            content = re.sub(pattern, replacement, content)
        
        # Remove volumes: block but keep networks: block
        # The sed command deletes everything between volumes: and networks: except networks: itself
        lines = content.split('\n')
        new_lines = []
        in_volumes_block = False
        
        for line in lines:
            if re.match(r'^volumes:', line):
                in_volumes_block = True
                continue  # Skip the volumes: line
            elif re.match(r'^networks:', line):
                in_volumes_block = False
                new_lines.append(line)  # Keep networks: line
            elif not in_volumes_block:
                new_lines.append(line)
        
        content = '\n'.join(new_lines)
        
        # Write the modified content
        with open(input_file, 'w') as f:
            f.write(content)
        
        return True
        
    except IOError as e:
        print(f"Error reading/writing {input_file}: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error modifying docker-compose file: {e}", file=sys.stderr)
        return False


def show_git_diff(input_file: str) -> None:
    """Show git diff of the modified file."""
    try:
        subprocess.run(['git', '-P', 'diff', input_file], check=False)
    except subprocess.CalledProcessError:
        print("Note: git diff failed (file may not be in git repository)")
    except FileNotFoundError:
        print("Note: git not found in PATH")


def dump_episodic_memories(output_file: str) -> bool:
    """
    Dump all episodic memories from the memory server to a JSON file.
    
    Args:
        output_file: Path to the output JSON file
        
    Returns:
        True if successful, False otherwise
    """
    base_url = "http://localhost:8080"
    
    try:
        # Step 1: Get all projects
        print(f"Fetching list of projects from {base_url}/api/v2/projects/list...")
        projects_resp = requests.post(
            f"{base_url}/api/v2/projects/list",
            headers={"accept": "application/json"},
            json={},
            timeout=30
        )
        projects_resp.raise_for_status()
        projects = projects_resp.json()
        
        if not isinstance(projects, list):
            print(f"Error: Unexpected response format from projects/list: {projects}", file=sys.stderr)
            return False
        
        print(f"Found {len(projects)} project(s)")
        
        # Structure to store all episodic memories organized by org_id and project_id
        all_memories = {}
        
        # Step 2: For each project, get all episodic memories
        for project in projects:
            org_id = project.get("org_id")
            project_id = project.get("project_id")
            
            if not org_id or not project_id:
                print(f"Warning: Skipping project with missing org_id or project_id: {project}", file=sys.stderr)
                continue
            
            print(f"Fetching episodic memories for org_id={org_id}, project_id={project_id}...")
            
            # Get episode count (optional, for progress tracking)
            try:
                count_resp = requests.post(
                    f"{base_url}/api/v2/projects/episode_count/get",
                    headers={"accept": "application/json"},
                    json={
                        "org_id": org_id,
                        "project_id": project_id
                    },
                    timeout=30
                )
                count_resp.raise_for_status()
                count_data = count_resp.json()
                episode_count = count_data.get("count", 0)
                print(f"  Found {episode_count} episode(s)")
            except Exception as e:
                print(f"  Warning: Could not get episode count: {e}")
                episode_count = None
            
            # Fetch all episodes with pagination
            all_episodes = []
            page_size = 100
            page_num = 0
            
            while True:
                try:
                    memories_resp = requests.post(
                        f"{base_url}/api/v2/memories/list",
                        headers={"accept": "application/json"},
                        json={
                            "org_id": org_id,
                            "project_id": project_id,
                            "page_size": page_size,
                            "page_num": page_num,
                            "filter": "",
                            "type": "episodic"
                        },
                        timeout=60
                    )
                    memories_resp.raise_for_status()
                    memories_data = memories_resp.json()
                    
                    content = memories_data.get("content", {})
                    episodic_memory = content.get("episodic_memory", [])
                    
                    if not episodic_memory:
                        break
                    
                    all_episodes.extend(episodic_memory)
                    print(f"  Fetched page {page_num}: {len(episodic_memory)} episode(s)")
                    
                    # If we got fewer episodes than page_size, we've reached the end
                    if len(episodic_memory) < page_size:
                        break
                    
                    page_num += 1
                    
                except requests.exceptions.RequestException as e:
                    print(f"  Error fetching page {page_num}: {e}", file=sys.stderr)
                    break
            
            # Store episodes organized by org_id and project_id
            if org_id not in all_memories:
                all_memories[org_id] = {}
            all_memories[org_id][project_id] = all_episodes
            print(f"  Total episodes for {org_id}/{project_id}: {len(all_episodes)}")
        
        # Step 3: Write to file
        print(f"Writing episodic memories to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(all_memories, f, indent=2)
        
        # Print summary
        total_episodes = sum(
            len(episodes)
            for org_data in all_memories.values()
            for episodes in org_data.values()
        )
        print(f"Successfully dumped {total_episodes} episodic memory episode(s) from {len(projects)} project(s) to {output_file}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with memory server: {e}", file=sys.stderr)
        return False
    except IOError as e:
        print(f"Error writing to file {output_file}: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error during dump: {e}", file=sys.stderr)
        return False


def upload_episodic_memories(input_file: str) -> bool:
    """
    Upload episodic memories from a JSON file to the memory server.
    
    Args:
        input_file: Path to the input JSON file
        
    Returns:
        True if successful, False otherwise
    """
    base_url = "http://localhost:8080"
    
    try:
        # Read the episodic memories file
        print(f"Reading episodic memories from {input_file}...")
        with open(input_file, 'r') as f:
            all_memories = json.load(f)
        
        if not isinstance(all_memories, dict):
            print(f"Error: Invalid file format. Expected a dictionary with org_id keys.", file=sys.stderr)
            return False
        
        total_uploaded = 0
        total_errors = 0
        
        # Iterate through each org_id and project_id
        for org_id, projects in all_memories.items():
            if not isinstance(projects, dict):
                print(f"Warning: Invalid format for org_id '{org_id}'. Expected a dictionary with project_id keys.", file=sys.stderr)
                continue
            
            for project_id, episodes in projects.items():
                if not isinstance(episodes, list):
                    print(f"Warning: Invalid format for {org_id}/{project_id}. Expected a list of episodes.", file=sys.stderr)
                    continue
                
                if not episodes:
                    print(f"Skipping {org_id}/{project_id}: no episodes to upload")
                    continue
                
                print(f"Uploading {len(episodes)} episode(s) for org_id={org_id}, project_id={project_id}...")
                
                # Convert episodes to messages format expected by API
                messages = []
                for episode in episodes:
                    if not isinstance(episode, dict):
                        print(f"  Warning: Skipping invalid episode (not a dict): {episode}", file=sys.stderr)
                        continue
                    
                    # Map episode fields to API message format
                    message = {
                        "content": episode.get("content", ""),
                        "producer": episode.get("producer_id", ""),
                        "produced_for": episode.get("produced_for_id", ""),
                        "timestamp": episode.get("created_at", ""),
                        "role": episode.get("producer_role", ""),
                        "metadata": episode.get("metadata") if episode.get("metadata") is not None else {}
                    }
                    messages.append(message)
                
                if not messages:
                    print(f"  Warning: No valid messages to upload for {org_id}/{project_id}")
                    continue
                
                # Upload to memory server
                try:
                    upload_resp = requests.post(
                        f"{base_url}/api/v2/memories",
                        headers={"accept": "application/json"},
                        json={
                            "org_id": org_id,
                            "project_id": project_id,
                            "types": ["episodic"],
                            "messages": messages
                        },
                        timeout=60
                    )
                    
                    if upload_resp.status_code == 200:
                        result = upload_resp.json()
                        results = result.get("results", [])
                        uploaded_count = len(results)
                        total_uploaded += uploaded_count
                        print(f"  Successfully uploaded {uploaded_count} episode(s)")
                    else:
                        print(f"  Error uploading to {org_id}/{project_id}: HTTP {upload_resp.status_code} - {upload_resp.text}", file=sys.stderr)
                        total_errors += len(messages)
                        
                except requests.exceptions.RequestException as e:
                    print(f"  Error uploading to {org_id}/{project_id}: {e}", file=sys.stderr)
                    total_errors += len(messages)
                except Exception as e:
                    print(f"  Unexpected error uploading to {org_id}/{project_id}: {e}", file=sys.stderr)
                    total_errors += len(messages)
        
        # Print summary
        print(f"\nUpload complete: {total_uploaded} episode(s) uploaded successfully")
        if total_errors > 0:
            print(f"Warning: {total_errors} episode(s) failed to upload", file=sys.stderr)
            # Continue processing but indicate partial failure
            return False
        
        return True
        
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}", file=sys.stderr)
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file {input_file}: {e}", file=sys.stderr)
        return False
    except IOError as e:
        print(f"Error reading file {input_file}: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error during upload: {e}", file=sys.stderr)
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Modify docker-compose.yml with Dropbox-aware volume path handling"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="lock",
        choices=["lock", "teardown", "upload_episodic_memories", "dump_episodic_memories"],
        help="Command to execute: 'lock' (default) or 'teardown' to delete lockfolder"
    )
    parser.add_argument(
        "filename",
        nargs="?",
        help="Filename for dump_episodic_memories or upload_episodic_memories commands"
    )
    
    args = parser.parse_args()
    
    # Handle commands that don't require Dropbox setup
    if args.command == "dump_episodic_memories":
        if not args.filename:
            print("Error: filename argument required for dump_episodic_memories", file=sys.stderr)
            sys.exit(1)
        print("Dumping episodic memories to local file...")
        if not dump_episodic_memories(args.filename):
            sys.exit(1)
        sys.exit(0)
    
    if args.command == "upload_episodic_memories":
        if not args.filename:
            print("Error: filename argument required for upload_episodic_memories", file=sys.stderr)
            sys.exit(1)
        # Upload might need Dropbox, but check later in the function
        print("Uploading episodic memories to Dropbox...")
        if not upload_episodic_memories(args.filename):
            sys.exit(1)
        sys.exit(0)
    
    # Commands below require Dropbox setup
    if not check_dropbox_data_directory():
        print("Error: DROPBOX_DATA_DIR environment variable not set. It should be the full path to the Dropbox directory where you want to host the postgres & neo4j files. Set using 'export DROPBOX_DATA_DIR=<path>' and try again.")
        sys.exit(1)

    if not check_dropbox_api_token():
        print("Error: DROPBOX_ACCESS_TOKEN environment variable not set. Set using 'export DROPBOX_ACCESS_TOKEN=<token>' and try again.")
        sys.exit(1)
    
    if args.command == "teardown":
        print("Tearing down Dropbox lockfolder...")
        lock_exists, lock_owned = check_dropbox_lockfile()
        if lock_exists and lock_owned:
            print("Lockfile found and owned by the current host; tearing down Dropbox lockfolder")
            if not release_dropbox_lock():
                print("Error: failed to release Dropbox lockfolder")
                sys.exit(1)
            else:
                print("Successfully released Dropbox lockfolder")
                sys.exit(0)
        else:
            print("Lockfile found but not owned by the current host; nothing to tear down, exiting")
            sys.exit(0)   
    
    # Normal execution
    print("Starting memmachine-dropbox script...")
    
    # Get user IDs
    uid, gid = get_user_ids()
    print(f"Using UID: {uid}, GID: {gid}")
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found", file=sys.stderr)
        sys.exit(1)

    volumes_path = VOLUMES_DIR

    # Attempt to acquire Dropbox lock
    lock_acquired, lock_error = attempt_dropbox_lock()

    if not lock_acquired:
        print(f"Lock acquisition failed: {lock_error}")
        print("Falling back to local copy")
        if not copy_volumes_to_local():
            print("Error: Failed to create local volumes copy", file=sys.stderr)
            sys.exit(1)
        volumes_path = VOLUMES_DIR_LOCAL

    # Ensure volume directories exist
    if not ensure_volume_directories(volumes_path):
        print("Error: Failed to create volume directories", file=sys.stderr)
        sys.exit(1)

    # Modify docker-compose.yml
    print(f"Modifying {INPUT_FILE} with volumes path: {volumes_path}")
    if not modify_docker_compose(INPUT_FILE, volumes_path, uid, gid):
        print(f"Error: Failed to modify {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Successfully wrote to {INPUT_FILE}")
    print("Showing diff:")
    time.sleep(1)
    show_git_diff(INPUT_FILE)


if __name__ == "__main__":
    main()

