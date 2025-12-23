#!/usr/bin/env bash

docker stop memmachine-postgres
docker stop memmachine-neo4j

while python3 ~/dropbox.py filestatus ${DROPBOX_DATA_DIR} | grep -q "sync"; do
    echo "Dropbox is still syncing, don't load on remote yet..."
    sleep 1
done

echo "Dropbox is synced! Go ahead and start on remote."
echo ":)"
