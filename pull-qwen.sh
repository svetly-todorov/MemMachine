#!/bin/sh
set -e
ollama serve &
sleep 3
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
wait
