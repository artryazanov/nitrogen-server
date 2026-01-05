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
# Check arguments passed to the script
if [ $# -eq 0 ]; then
    echo "Starting server with default configuration..."
    exec python scripts/serve.py "$MODEL_FILE"
else
    # Check if the first argument starts with "-" (flag)
    # If it is a flag, we assume the user wants to use the default model and append flags.
    if [[ "$1" == -* ]]; then
        echo "Starting server with default model and flags: $@"
        exec python scripts/serve.py "$MODEL_FILE" "$@"
    else
        # If the first argument does NOT start with "-", we assume it is a custom model path.
        # In this case, we pass all arguments directly to serve.py without injecting the default model.
        echo "Starting server with custom arguments: $@"
        exec python scripts/serve.py "$@"
    fi
fi
