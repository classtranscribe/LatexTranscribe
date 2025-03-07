import requests
from io import BytesIO
from zipfile import ZipFile
url = "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip"
file_name = "english_g2.zip"
timeout = 1000
r = requests.get(url, timeout=timeout)
if r.ok:
    with ZipFile(BytesIO(r.content)) as zip_ref:
        zip_ref.extractall("models/EasyOCR")