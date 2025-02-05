from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from PIL import Image
import io

from src.pipeline import Pipeline
from main import load_config

app = FastAPI()

config_path = "./configs/config.yaml"
pipeline = Pipeline(load_config(config_path))

def process_image(image: Image.Image) -> Image.Image:
    """Example image processing function (grayscale conversion)."""
    return image.convert("L")

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Read image bytes from the uploaded file
        image_bytes = await file.read()

        # Convert the bytes to a PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Process the image
        processed_image = process_image(image)
        # processed_image.save("./test.png")

        # Convert processed image back to bytes
        img_bytes = io.BytesIO()
        processed_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return Response(content=img_bytes.read(), media_type="image/png")
    
    except Exception as e:
        return Response(status=500)
