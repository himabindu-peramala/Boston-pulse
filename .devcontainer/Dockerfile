FROM --platform=linux/amd64 python:3.10

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install TFDV for data validation
RUN pip install tensorflow-data-validation

CMD ["bash"]
