#!/usr/bin/env bash
set -euo pipefail

UID_VAL=$(id -u)
GID_VAL=$(id -g)

INPUT="docker-compose.yml"
# OUTPUT="docker-compose.generated.yml"

mkdir -p docker_volumes/{postgres_data,neo4j_data,neo4j_logs,neo4j_import,neo4j_plugins,memmachine_logs}

# Replace named volumes with bind mounts
# Remove existing user: lines (postgres + neo4j only)
# Insert correct user after container_name
# Remove only the volumes: block, keep networks: block
sed -i \
  -e 's|^\(\s*-\s*\)postgres_data:|\1./docker_volumes/postgres_data:|' \
  -e 's|^\(\s*-\s*\)neo4j_data:|\1./docker_volumes/neo4j_data:|' \
  -e 's|^\(\s*-\s*\)neo4j_logs:|\1./docker_volumes/neo4j_logs:|' \
  -e 's|^\(\s*-\s*\)neo4j_import:|\1./docker_volumes/neo4j_import:|' \
  -e 's|^\(\s*-\s*\)neo4j_plugins:|\1./docker_volumes/neo4j_plugins:|' \
  -e 's|^\(\s*-\s*\)memmachine_logs:|\1./docker_volumes/memmachine_logs:|' \
  -e "/^  postgres:/,/^  [a-z]/{ /^    user:/d }" \
  -e "/^  neo4j:/,/^  [a-z]/{ /^    user:/d }" \
  -e "/container_name: memmachine-postgres/a\\
    user: \"${UID_VAL}:${GID_VAL}\"" \
  -e "/container_name: memmachine-neo4j/a\\
    user: \"${UID_VAL}:${GID_VAL}\"" \
  -e '/^volumes:/,/^networks:/{
        /^networks:/!d
      }' \
  "$INPUT"

echo "Wrote to $INPUT"
echo "Showing diff:"
sleep 1
git diff $INPUT

