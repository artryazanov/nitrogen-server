# NitroGen Server

[![Python application](https://github.com/artryazanov/nitrogen-server/actions/workflows/python-app.yml/badge.svg)](https://github.com/artryazanov/nitrogen-server/actions/workflows/python-app.yml)
[![License: NVIDIA](https://img.shields.io/badge/License-NVIDIA-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

NitroGen Server is a specialized inference server for the **NitroGen** foundation model (originally by MineDojo). It provides a high-performance backend for generalist gaming agents, allowing them to play games by processing visual input and generating controller commands.

This project dockerizes the original NitroGen implementation and extends it with a **dual-protocol architecture**, enabling connections from both Python-based clients and external tools like **BizHawk (Lua)**.

> [!IMPORTANT]
> **License & Usage Restrictions**: This project is based on NVIDIA's Work and is licensed under the NVIDIA License. It is strictly for non-commercial research purposes only. Use for military, surveillance, nuclear technology, or biometric processing is expressly prohibited.

## ✨ Features

*   **Foundation Model**: powered by `nvidia/NitroGen`, a large multimodal model for game control.
*   **Dual Protocol Support**:
    *   **ZeroMQ + Pickle**: Fast, native communication for Python clients.
    *   **TCP + JSON**: Universal standard for connecting from Lua, C#, or other languages (perfect for Emulator integration).
*   **Dockerized**: Zero-dependency deployment on the host. Handles CUDA drivers and environment setup automatically.
*   **Auto-Healing**: Automatically downloads the model weights (`ng.pt`) on the first run if they are missing.
*   **Persistent Caching**: Uses a local volume for models to avoid re-downloading.
*   **LoRA Support**: Use custom fine-tuned LoRA adapters with auto-merging capabilities.

---

## 🚀 Quick Start (Docker)

This is the recommended way to run the server.

### Prerequisites

*   Host with an NVIDIA GPU (Linux or Windows)
*   [Docker](https://docs.docker.com/engine/install/) & Docker Compose
*   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (required for GPU access)

### 1. Clone Repository
```bash
git clone https://github.com/artryazanov/nitrogen-server.git
cd nitrogen-server
```

### 2. Start the Server
```bash
docker-compose up --build
```

**On the first run**, the server will automatically download the ~2GB model checkpoint. This may take a few minutes.
Once ready, you will see:
```text
ZMQ Server running on port 5555
Simple TCP Server (JSON+Bytes) running on port 5556
```

---

## 💻 Usage

### Connecting Clients

The server exposes two ports by default:

| Protocol | Port | Description | Target Use Case |
| :--- | :--- | :--- | :--- |
| **ZeroMQ** | `5555` | Serialized Python objects (Pickle) | **Python Clients** (e.g., `scripts/play.py`) |
| **TCP/JSON** | `5556` | JSON Header + Image (BMP/PNG) or Raw Bytes | **BizHawk / Emulators** / Non-Python |

### Python Client Example
We provide a `play.py` script to connect a game running on a client (e.g. Windows) to the NitroGen server.

```bash
# On your Windows Gaming Machine
python scripts/play.py --process "celeste.exe" --ip <SERVER_IP> --port 5555
```

### BizHawk (Lua) Integration
For emulators like BizHawk, use the TCP protocol on port **5556**.

We provide a ready-to-use client script in a separate repository: [NitroGen BizHawk AI Agent](https://github.com/artryazanov/nitrogen-bizhawk-ai-agent).

Please refer to the [repository documentation](https://github.com/artryazanov/nitrogen-bizhawk-ai-agent#readme) for setup and usage instructions.

#### Protocol Details
The server supports **Automatic Format Detection** on port **5556**. It handles **Any Image Format** (PNG, BMP, JPG) or **Raw Pixels**.

Steps:
1.  **Open Socket**: Connect to `<SERVER_IP>:5556`.
2.  **Send Request**:
    Send a JSON header terminated by `\n`. It is **highly recommended** to include the `len` field (file size in bytes) to ensure perfect synchronization.

    ```json
    {
        "type": "predict",
        "len": 12345
    {
        "type": "predict",
        "len": 12345,
        "resize_mode": "pad"
    }
    ```
    
    **Resize Modes (`resize_mode`):**
    *   `pad` (Default): Pads the image with black borders to preserve aspect ratio (adds bars), then resizes to 256x256.
    *   `crop`: Center-crops a square from the image, then resizes to 256x256.
    *   `stretch`: Stretches the image to fit 256x256 (may distort aspect ratio).
3.  **Send Image**:
    *   **Option A (Recommended):** Send a standard image file (PNG, BMP, JPG). The server uses `cv2.imdecode` to parse it automatically.
    *   **Option B (Fallback):** Send **196,608 bytes** of raw RGB pixel data (256x256). If `len` matches exactly, it is treated as raw buffer.
4.  **Receive Response**: Read the JSON response terminated by `\n`.

---

## 🐞 Debugging Mode

You can enable debug mode to save detailed artifacts for every request (received image, JSON parameters, processed image, model response).

**Enable via CLI (Manual):**
```bash
python scripts/serve.py models/nvidia/NitroGen/ng.pt --debug --debug-dir debug_output
```

**Enable via Docker:**
To run with debug mode in Docker, use `docker-compose run` to pass the flag and map the ports:
```bash
docker-compose run --service-ports nitrogen-server --debug
```
*Note: This will output artifacts to the `debug/` folder on your host machine (mapped in docker-compose.yml).*

**Artifacts generated:**
1. `*_1_received.png`: The original image received from the client.
2. `*_2_params.json`: The JSON parameters of the request.
3. `*_3_processed.png`: The preprocessed image (resized/padded) sent to the model.
4. `*_4_response.json`: The model's prediction response.


**Run with LoRA Adapter:**
To use a different model (like a LoRA adapter) with Docker, use `docker-compose run` to override the start command arguments:

```bash
# Ensure your checkpoint is in the `models/` directory (e.g. models/checkpoints/final_model)
docker-compose run --service-ports nitrogen-server models/checkpoints/final_model --base-model models/nvidia/NitroGen/ng.pt
```

---

## 🛠 Manual Installation (Development)

If you prefer to run the server without Docker (e.g., for development):

```bash
# 1. Install dependencies
pip install -e .[serve] peft
pip install "huggingface_hub[cli]"

# 2. Download Model
# The server requires the model weights (~2GB) to be downloaded locally.
huggingface-cli download nvidia/NitroGen ng.pt --local-dir models/nvidia/NitroGen

# 3. Run Server
python scripts/serve.py models/nvidia/NitroGen/ng.pt [--debug]

# 4. Run with LoRA Adapter
# Point to the LoRA directory. Ensure the base model is also available.
python scripts/serve.py models/checkpoints/final_model --base-model models/nvidia/NitroGen/ng.pt
```

## 📂 Project Structure

*   `nitrogen/`: Core library code (model definition, inference logic).
*   `scripts/`: Executable scripts.
    *   `serve.py`: The main server entry point.
    *   `play.py`: Python client script for running agents.
    *   `start.sh`: Entrypoint script for Docker.
*   `models/`: Directory for storing downloaded model weights (gitignored).
*   `tests/`: Unit and integration tests.
*   `Dockerfile`: Definition for the server container.

## 🔗 Credits

This project is a fork and extension of the original work by **MineDojo**.
*   **Original Repository**: [MineDojo/NitroGen](https://github.com/MineDojo/NitroGen)
*   **Hugging Face Model**: [nvidia/NitroGen](https://huggingface.co/nvidia/NitroGen)

Check [README_ORIGINAL.md](README_ORIGINAL.md) for the original documentation.
