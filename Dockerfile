# Use the official Python image as the base
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Clone the GenmoAI models repository
RUN git clone https://github.com/genmoai/models.git .

# Install Python dependencies
RUN pip install --no-cache-dir uv setuptools
RUN uv venv .venv
RUN . .venv/bin/activate && uv pip install -e . --no-build-isolation

# Download model weights
RUN mkdir -p /app/weights && \
    curl -L -o /app/weights/mochi-1-preview.pth https://huggingface.co/genmo/mochi-1-preview/resolve/main/pytorch_model.bin

# Set environment variables
ENV MODEL_DIR=/app/weights

# Expose the port for the Gradio interface
EXPOSE 7860

# Command to run the Gradio UI
CMD ["/bin/bash", "-c", ". .venv/bin/activate && python3 -m mochi_preview.gradio_ui --model_dir $MODEL_DIR"]
