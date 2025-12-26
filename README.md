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

You can download the script directly here: [`bizhawk_ai_agent.lua`](https://github.com/artryazanov/nitrogen-bizhawk-ai-agent/blob/main/bizhawk_ai_agent.lua).

Please refer to the [repository documentation](https://github.com/artryazanov/nitrogen-bizhawk-ai-agent#readme) for setup and usage instructions.

#### Protocol Details
The server now supports **Automatic Format Detection** on port **5556**. It handles **Any Image Format** (PNG, BMP, JPG) or **Raw Pixels**.

Steps:
1.  **Open Socket**: Connect to `<SERVER_IP>:5556`.
2.  **Send Request**:
    Send a JSON header terminated by `\n`. It is **highly recommended** to include the `len` field (file size in bytes) to ensure perfect synchronization.

    ```json
    {
        "type": "predict",
        "len": 12345
    }
    ```
3.  **Send Image**:
    *   **Option A (Recommended):** Send a standard image file (PNG, BMP, JPG). The server uses `cv2.imdecode` to parse it automatically.
    *   **Option B (Fallback):** Send **196,608 bytes** of raw RGB pixel data (256x256). If `len` matches exactly, it is treated as raw buffer.
4.  **Receive Response**: Read the JSON response terminated by `\n`.

---

## 🛠 Manual Installation (Development)

If you prefer to run the server without Docker (e.g., for development):

```bash
# 1. Install dependencies
pip install -e .[serve]
pip install "huggingface_hub[cli]"

# 2. Download Model
# The server requires the model weights (~2GB) to be downloaded locally.
huggingface-cli download nvidia/NitroGen ng.pt --local-dir models/nvidia/NitroGen

# 3. Run Server
python scripts/serve.py models/nvidia/NitroGen/ng.pt
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
