# Use an official Python 3.10 image as the base image
FROM python:3.10-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    slurm-wlm \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install Python dependencies
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    triton==3.0.0 \
    transformers==4.46.3 \
    safetensors==0.4.5 \
    vllm \
    axolotl \
    deepspeed

# Set up Slurm (if needed)
# You may need to configure Slurm further depending on your setup
COPY slurm.conf /etc/slurm-llnl/slurm.conf

# Set the working directory
WORKDIR /workspace

# Copy your DeepSeek model and scripts into the container
COPY . /workspace

# Expose any necessary ports (if applicable)
# EXPOSE <port>

# Command to run your DeepSeek model with Slurm
CMD ["srun", "python", "your_deepseek_script.py"]