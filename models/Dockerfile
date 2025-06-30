FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY xgboost_model.pkl .
COPY preprocessing_info.pkl .
COPY xgboost_model_server.py .

# Set environment variables
ENV MODEL_NAME=xgboost-model
ENV API_TYPE=REST
ENV SERVICE_TYPE=MODEL
ENV SELDON_LOG_LEVEL=INFO

# Expose ports
EXPOSE 9000 5001

# Run the model server
CMD ["python", "xgboost_model_server.py"] 