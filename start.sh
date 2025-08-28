#!/bin/bash
set -e  # Exit on any error

# Function to kill background processes on exit
cleanup() {
    echo "Stopping all services..."
    kill $(jobs -p) 2>/dev/null || true
    exit
}
trap cleanup SIGINT SIGTERM

echo "Starting Ollama server..."
/bin/ollama serve &

echo "Waiting for Ollama to be ready..."
until curl -s http://localhost:11434 >/dev/null; do
    echo "Waiting for Ollama..."
    sleep 2
done

echo "Creating the model from Modelfile..."
/bin/ollama create granite2b -f /root/.ollama/models/Modelfile

echo "Starting FastAPI backend..."
# Start the FastAPI server with Uvicorn
cd /backend
uvicorn main:app --host 0.0.0.0 --port 8000 &

echo "Waiting for FastAPI to start..."
until curl -s http://localhost:8000 >/dev/null; do
    sleep 1
done

echo "Starting Streamlit frontend..."
# Start Streamlit in the foreground (this will block, keeping the container alive)
cd /frontend
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.baseUrlPath=/ &

# Wait for all background processes. If any crash, the script exits.
wait -n