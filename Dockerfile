FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (for tokenizers)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Set workdir
WORKDIR /app
COPY . /app

# Install Python deps
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
