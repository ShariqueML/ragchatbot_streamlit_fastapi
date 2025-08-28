# FROM ollama/ollama:0.11.3-rc0-rocm
FROM ollama/ollama:latest

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/granite32b.gguf /root/.ollama/models/
COPY modelfiles/Modelfile /root/.ollama/models/

COPY backend/ ./backend/
COPY frontend/ ./frontend/

COPY start.sh /start.sh
RUN chmod +x /start.sh

# HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
#   CMD curl -f http://0.0.0.0:11434 || exit 1
  
# Expose the port that will be publicly accessed (Streamlit)
EXPOSE 8501

# Use the custom startup script
CMD ["/start.sh"]