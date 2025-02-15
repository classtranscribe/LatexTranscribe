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
        "models/MFR/*",
        "models/TabRec/*",
    ],
)
print("Download complete.")
