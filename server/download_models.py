# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface-hub[hf-transfer]",
# ]
# ///
from huggingface_hub import snapshot_download

print("Downloading models...")
snapshot_download(
    repo_id="opendatalab/pdf-extract-kit-1.0",
    local_dir="./",
    allow_patterns=[
        "models/Layout/*",
        "models/MFD/*",
        "models/MFR/unimernet_base/*",
        "models/TabRec/*",
    ],
)

import requests
from io import BytesIO
from zipfile import ZipFile
urls = [
    "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip",
    "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip"
]
for url in urls:
    file_name = "english_g2.zip"
    timeout = 1000
    r = requests.get(url, timeout=timeout)
    if r.ok:
        with ZipFile(BytesIO(r.content)) as zip_ref:
            zip_ref.extractall("models/EasyOCR")

print("Download complete.")
