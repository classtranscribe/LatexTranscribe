FROM python:3.11-slim-bookworm

# Install OS dependencies vim-tiny  netcat-openbsd 
# git and clang for detectron2
RUN apt-get -qq update && \
    apt-get -qq install --no-install-recommends  git  clang && \
    apt-get -qq clean autoclean && \
    apt-get -qq autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip
    
# Install CUDA  (and other setup tools); This next line is based on Dockerfile in  https://github.com/jim60105/docker-whisperX/
# docker-whisperX needs numpy<2 do we? (see # https://github.com/jim60105/docker-whisperX/issues/40)
#     # torch==2.2.2 torchaudio==2.2.2 \
#     pyannote.audio==3.1.1 \
#     "numpy<2.0"
# RUN pip --force-reinstall pip setuptools wheel
#    pip install -U --extra-index-url https://download.pytorch.org/whl/cu126 

# detectron needs PyTorch ≥ 1.8 and companion torchvision
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 
RUN python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# https://detectron2.readthedocs.io/en/latest/tutorials/install.html
# It takes 37s to build detectron on my M1 mac
RUN git clone https://github.com/facebookresearch/detectron2.git
RUN CC=clang CXX=clang++ python -m pip install -e detectron2

RUN pip install fastapi uvicorn
  
COPY . .  

CMD ["uvicorn" , "helloworld:app" , "--host", "0.0.0.0"]
  
  
  
    
#COPY requirements-linux-gpudocker-build.txt .
#RUN pip install --no-cache-dir -r requirements-linux-gpudocker-build.txt

#COPY . .

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Making a note for later, For M1 support docker-whisperX also includes 
# apt-get update && apt-get install -y --no-install-recommends libgomp1=12.2.0-14 libsndfile1=1.2.0-1;
