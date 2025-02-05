FROM --platform=linux/amd64 python:3.11-slim-bookworm

# Install OS dependencies
RUN apt-get -qq update && \
    apt-get -qq install --no-install-recommends vim-tiny  netcat-openbsd && \
    apt-get -qq clean autoclean && \
    apt-get -qq autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip
    
# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "FIXME" ]
