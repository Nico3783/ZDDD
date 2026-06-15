FROM python:3.13-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY config/ config/
COPY models/ models/
COPY scripts/ scripts/
COPY data/ data/

# Install package
RUN pip install --no-cache-dir -e .

# Expose dashboard port
EXPOSE 8501

# Default: run the dashboard
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
