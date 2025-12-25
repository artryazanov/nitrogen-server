# NitroGen Server

NitroGen is an open foundation model for generalist gaming agents. This repository hosts the inference server which can be easily deployed using Docker.

For the original documentation, including manual installation steps and details on the client-side game agent, please refer to [README_ORIGINAL.md](README_ORIGINAL.md) or the [original repository](https://github.com/MineDojo/NitroGen).

## Features

- **Dockerized Server**: Easy deployment with Docker and Docker Compose.
- **Auto-Model Download**: The server automatically downloads the `nvidia/NitroGen` model if it's missing.
- **Persistent Storage**: Models are saved to a local volume to avoid re-downloading.
- **GPU Support**: Configured for NVIDIA GPUs out of the box.

## Prerequisites

- **Docker** and **Docker Compose**
- **NVIDIA Container Toolkit**: Required for GPU access inside the container. [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/artryazanov/NitroGen-Server.git
   cd NitroGen-Server
   ```

2. **Start the server:**
   ```bash
   docker-compose up --build
   ```

   - On the first run, the container will download the model (`ng.pt`) to `models/nvidia/NitroGen/`. This may take some time depending on your internet connection.
   - Subsequent runs will use the cached model.
   - The server listens on port **5555**.

3. **Check Status:**
   You should see output indicating the server is running:
   ```
   Server running on port 5555
   Waiting for requests...
   ```

## Configuration

### Volumes
The `docker-compose.yml` mounts a local directory for model persistence:
- **Local:** `./models/nvidia/NitroGen`
- **Container:** `/app/models/nvidia/NitroGen`

This ensures that the large model checkpoint is stored on your host machine and persists across container restarts.

### Ports
- **5555**: The ZeroMQ server port. You can change this in `docker-compose.yml` if needed.

## Connecting a Client

Once the server is running (usually on a Linux machine with a powerful GPU), you can connect to it from your Windows gaming machine using `scripts/play.py`.

Ensure your client machine can reach the server's IP address on port 5555.

```bash
# On your Windows machine (Client)
python scripts/play.py --process '<game_executable_name>.exe' --ip <SERVER_IP> --port 5555
```

*(Note: You might need to modify `play.py` to accept IP/Port arguments if it hardcodes localhost, or use SSH tunneling).*

## Credits & Acknowledgements

This project is a fork of [NitroGen](https://github.com/MineDojo/NitroGen) by MineDojo. I extend my sincere gratitude to the original authors for their groundbreaking work in creating an open foundation model for generalist gaming agents.
