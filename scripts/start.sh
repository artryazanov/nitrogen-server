#!/bin/bash
set -e

MODEL_DIR="models/nvidia/NitroGen"
MODEL_FILE="$MODEL_DIR/ng.pt"

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Check if model exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "Model ng.pt not found in $MODEL_DIR. Downloading from nvidia/NitroGen..."
    # Install huggingface_hub if not present (should be in docker, but good check)
    if ! command -v huggingface-cli &> /dev/null; then
        echo "Error: huggingface-cli not found. Please ensure huggingface_hub[cli] is installed."
        exit 1
    fi
    
    # Download the model file
    # We use 'hf' alias if available, or fall back to huggingface-cli if needed, 
    # but the warning suggests 'hf'.
    # Also removing --local-dir-use-symlinks as it's deprecated/ignored for local-dir.
    hf download nvidia/NitroGen ng.pt --local-dir "$MODEL_DIR"
    echo "Download complete."
else
    echo "Model ng.pt found at $MODEL_FILE."
fi

# Pass specific arguments to serve.py
# If no arguments provided, use defaults but always pass the checkpoint
if [ $# -eq 0 ]; then
    echo "Starting server with default configuration..."
    exec python scripts/serve.py "$MODEL_FILE"
else
    # If arguments are provided, user might have wanted to specify flags.
    # However, serve.py requires 'ckpt' as the first positional argument.
    # We should assume the user passes flags, and we prepend the model file?
    # Or we assume the user overrides everything?
    # The requirement is "Server ... should download ... and save ...".
    # It implies the container manages the model path.
    
    echo "Starting server with arguments: $@"
    exec python scripts/serve.py "$MODEL_FILE" "$@"
fi
