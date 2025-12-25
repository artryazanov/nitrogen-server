# NitroGen Server

NitroGen Server is a specialized inference server for the **NitroGen** foundation model (originally by MineDojo). It provides a high-performance backend for generalist gaming agents, allowing them to play games by processing visual input and generating controller commands.

This project dockerizes the original NitroGen implementation and extends it with a **dual-protocol architecture**, enabling connections from both Python-based clients and external tools like **BizHawk (Lua)**.

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

*   Linux Host with an NVIDIA GPU
*   [Docker](https://docs.docker.com/engine/install/) & Docker Compose
*   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (required for GPU access)

### 1. Clone & Rename
```bash
git clone https://github.com/artryazanov/NitroGen-Server.git
cd NitroGen-Server
```

### 2. Start the Server
```bash
docker-compose up --build
```

**On the first run**, the server will automatically download the ~7GB model checkpoint. This may take a few minutes.
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
| **TCP/JSON** | `5556` | JSON Header + Raw Bytes | **BizHawk / Emulators** / Non-Python |

### Python Client Example
We provide a `play.py` script to connect a game running on a Windows client to the Linux server.

```bash
# On your Windows Gaming Machine
python scripts/play.py --process "celeste.exe" --ip <SERVER_IP> --port 5555
```

### BizHawk (Lua) Integration
For emulators like BizHawk, use the TCP protocol on port **5556**.

1.  **Open Socket**: Connect to `<SERVER_IP>:5556`.
2.  **Send Request**: Send a JSON string terminated by `\n`.
    ```json
    {"type": "predict"}
    ```
3.  **Send Image**: Immediately send **196,608 bytes** of raw pixel data (256x256 RGB).
4.  **Receive Response**: Read the JSON response terminated by `\n`.

_See `scripts/serve.py` for the implementation details._

---

## 🛠 Manual Installation (Development)

If you prefer to run the server without Docker (e.g., for development):

```bash
# 1. Install dependencies
pip install -e .[serve]

# 2. Download Model (Automatic on first run, but you can pre-download)
# The server looks for models in ./models/nvidia/NitroGen/ng.pt

# 3. Run Server
python scripts/serve.py models/nvidia/NitroGen/ng.pt
```

## 📂 Project Structure

*   `nitrogen/`: Core library code (model definition, inference logic).
*   `scripts/`: Executable scripts.
    *   `serve.py`: The main server entry point.
    *   `play.py`: Client script for running agents.
*   `models/`: Directory for storing downloaded model weights (gitignored).
*   `Dockerfile`: Definition for the server container.

## 🔗 Credits

This project is a fork and extension of the original work by **MineDojo**.
*   **Original Repository**: [MineDojo/NitroGen](https://github.com/MineDojo/NitroGen)
*   **Hugging Face Model**: [nvidia/NitroGen](https://huggingface.co/nvidia/NitroGen)

Check [README_ORIGINAL.md](README_ORIGINAL.md) for the original documentation.
