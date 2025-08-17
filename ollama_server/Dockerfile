FROM ollama/ollama:0.11.3-rc0-rocm

COPY models/granite32b.gguf /root/.ollama_server/models/
COPY modelfiles/Modelfile /root/.ollama_server/models/
# Create the Ollama model
RUN ollama create granite2b -f /root/.ollama_server/models/Modelfile

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:11434 || exit 1
  
EXPOSE 11434
CMD ["ollama", "serve"]