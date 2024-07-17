

# Install needed packages on linux
```bash
sudo apt-get update && sudo apt-get install -y \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    python3.10-venv\
    && rm -rf /var/lib/apt/lists/*
```
# Create and activate virtual environment
```bash
python -m venv env
source env/bin/activate
```
# Install the following packages 
```bash
pip install -r requirements.backend.txt -r requirements.frontend.txt
```
# Create a .env file with your cloud keys
fill in the blank quotes to add the following attributes and save as .env file in the main repo

```
# OpenAI keys
OPENAI_API_KEY=""

# Azure keys
AZURE_SEARCH_SERVICE_NAME=""
AZURE_SEARCH_ADMIN_KEY=""
AZURE_SEARCH_INDEX_NAME=""
```

