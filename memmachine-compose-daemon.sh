#!/usr/bin/env bash

DROPBOX_VOLUMES_REQUESTER="${DROPBOX_DATA_DIR}/docker_volumes_owner"
DROPBOX_VOLUMES_LOCKFILE="${DROPBOX_DATA_DIR}/docker_volumes_lockfolder/lockfile.txt"

main() {
    local lock_owner=""
    local volume_requester=""

    echo "Checking for ${DROPBOX_VOLUMES_LOCKFILE} and ${DROPBOX_VOLUMES_REQUESTER}."
    while true; do
        if [[ ! -f "${DROPBOX_VOLUMES_LOCKFILE}" ]]; then
            sleep 1
            continue
        fi
        if [[ ! -f "${DROPBOX_VOLUMES_REQUESTER}" ]]; then
            sleep 1
            continue
        fi
        echo "${DROPBOX_VOLUMES_LOCKFILE} exists and ${DROPBOX_VOLUMES_REQUESTER} exists."
        break
    done

    echo "Watching ${DROPBOX_VOLUMES_REQUESTER} and ${DROPBOX_VOLUMES_LOCKFILE}."

    while true; do
        lock_owner="$(<"${DROPBOX_VOLUMES_LOCKFILE}")"
        volume_requester="$(<"${DROPBOX_VOLUMES_REQUESTER}")"
        if [[ "${volume_requester}" != "$(hostname)" ]]; then
            if [[ "${lock_owner}" == "$(hostname)" ]]; then
                echo ""
                echo "Docker volume lockfile is owned by current host ${lock_owner}, but Docker volume requester ${volume_requester} is no longer current host $(hostname)."
                echo "Therefore someone else wants the databases. Putting local memmachine containers to sleep..."
                break
            fi
        fi
        sleep 1
    done

    docker stop memmachine-postgres
    # docker stop memmachine-neo4j

    while python3 ~/dropbox.py filestatus ${DROPBOX_DATA_DIR} | grep -q "sync"; do
        echo "Dropbox is still syncing, don't load on remote yet..."
        sleep 1
    done

    echo "Tearing down dropbox lock..."
    while ! python3 memmachine-dropbox.py teardown; do
        echo "Failed to tear down dropbox lock. Please see errors. Waiting 10 seconds and retrying..."
        sleep 10
    done

    # Sleep for 10 seconds to allow the lock folder to disappear
    sleep 10
}

while true; do
    main
done