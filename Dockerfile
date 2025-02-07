FROM --platform=linux/amd64 python:3.11-slim-bookworm

# Install OS dependencies
RUN apt-get -qq update && \
    apt-get -qq install --no-install-recommends vim-tiny  netcat-openbsd && \
    apt-get -qq clean autoclean && \
    apt-get -qq autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip
    
# Install CUDA12.1 (and other setup tools); This next line is based on Dockerfile in  https://github.com/jim60105/docker-whisperX/
# docker-whisperX needs numpy<2 do we? (see # https://github.com/jim60105/docker-whisperX/issues/40)

RUN pip install -U --force-reinstall pip setuptools wheel && \
    pip install -U --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.2 torchaudio==2.2.2 \
    pyannote.audio==3.1.1 \    
    "numpy<2.0"

# Making a note for later, For M1 support docker-whisperX also includes 
# apt-get update && apt-get install -y --no-install-recommends libgomp1=12.2.0-14 libsndfile1=1.2.0-1;
    
    
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
