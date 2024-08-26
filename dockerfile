# FINRAGIFY_APP/Dockerfile

FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /FINRAGIFY_APP

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/apsinghAnalytics/FinRAGify_App.git .


# Before installing packages from requirements.txt, install the following light weight torch package from custom-URL

RUN pip3 install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Copy the .env file when the .env file is in same directory as dockerfile
COPY .env .env

# Expose Streamlit's default port
EXPOSE 8501

# Add a health check to verify the app is running
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Set the entry point to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]