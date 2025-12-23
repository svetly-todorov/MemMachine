#!/usr/bin/env bash

while python3 ~/dropbox.py filestatus ${DROPBOX_DATA_DIR} | grep -q "sync"; do
    echo "Dropbox is still syncing, don't start yet..."
    sleep 1
done

echo "Dropbox is synced! Starting database containers..."

docker start memmachine-postgres
docker start memmachine-neo4j
