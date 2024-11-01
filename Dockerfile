# Use the official NVIDIA CUDA image as the base
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH="/root/.local/bin:${PATH}"

# Install system dependencies and add Python 3.10 repository
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    ffmpeg \
    git \
    wget \
    curl \
    ca-certificates && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Create a symbolic link for pip3 if needed
RUN ln -sf /usr/local/bin/pip3 /usr/bin/pip


# Clone the GenmoAI models repository
RUN git clone https://github.com/genmoai/models /opt/models

# Set the working directory
WORKDIR /opt/models

# Create and activate a virtual environment
RUN pip install setuptools && \
    pip install -e . --no-build-isolation

# Download model weights
RUN mkdir -p /opt/models/weights && \
    python3 ./scripts/download_weights.py /opt/models/weights

# Expose the port for the Gradio UI
EXPOSE 7860

# Set the entrypoint to run the Gradio UI
ENTRYPOINT ["python3", "./demos/gradio_ui.py", "--model_dir", "/opt/models/weights"]
