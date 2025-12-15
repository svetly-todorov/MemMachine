#!/usr/bin/env bash
set -euo pipefail

UID_VAL=$(id -u)
GID_VAL=$(id -g)

INPUT="docker-compose.yml"
# OUTPUT="docker-compose.generated.yml"

mkdir -p docker_volumes/{postgres_data,neo4j_data,neo4j_logs,neo4j_import,neo4j_plugins,memmachine_logs}

sed -i \
  -e "s|postgres_data:/var/lib/postgresql/data|./docker_volumes/postgres_data:/var/lib/postgresql/data|g" \
  -e "s|neo4j_data:/data|./docker_volumes/neo4j_data:/data|g" \
  -e "s|neo4j_logs:/logs|./docker_volumes/neo4j_logs:/logs|g" \
  -e "s|neo4j_import:/var/lib/neo4j/import|./docker_volumes/neo4j_import:/var/lib/neo4j/import|g" \
  -e "s|neo4j_plugins:/plugins|./docker_volumes/neo4j_plugins:/plugins|g" \
  -e "s|memmachine_logs:/tmp/memory_logs|./docker_volumes/memmachine_logs:/tmp/memory_logs|g" \
  -e "/container_name: memmachine-postgres/a\\
    user: \"${UID_VAL}:${GID_VAL}\"" \
  -e "/container_name: memmachine-neo4j/a\\
    user: \"${UID_VAL}:${GID_VAL}\"" \
  -e "/^volumes:/,\$d" \
  "$INPUT"

echo "Wrote $OUTPUT"

