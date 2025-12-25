FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
# libgl1 and libglib2.0-0 are required for opencv-python
# git is just in case pip needs it for some packages
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy project configuration files
COPY pyproject.toml README.md /app/

# Copy application code (needed for pip install . to find the package)
COPY nitrogen /app/nitrogen
COPY scripts /app/scripts

# Install python dependencies
# We install the package in editable mode or just install the dependencies
# Installing [serve] extras and huggingface_hub for the download script
RUN pip install --upgrade pip && \
    pip install "huggingface_hub[cli]" && \
    pip install ".[serve]"

# Ensure start script is executable
RUN chmod +x /app/scripts/start.sh

# Ensure start script is executable
RUN chmod +x /app/scripts/start.sh

# Expose the default port
EXPOSE 5555

# Define volume for models
VOLUME /app/models/nvidia/NitroGen

# Set entrypoint
ENTRYPOINT ["/app/scripts/start.sh"]
