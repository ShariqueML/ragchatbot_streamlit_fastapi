#!/bin/sh

echo "Starting Ollama server..."
/bin/ollama serve &

# Wait for Ollama to be ready
until curl -s http://localhost:11434 >/dev/null; do
  echo "Waiting for Ollama to start..."
  sleep 2
done

echo "Pulling model: granite3.3"
/bin/ollama pull granite3.3

wait
