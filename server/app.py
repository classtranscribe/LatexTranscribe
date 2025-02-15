from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from PIL import Image
import io

from src.pipeline import Pipeline
from main import load_config

app = FastAPI()

# Allow CORS only from frontend running on localhost:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend only from localhost:5173
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config_path = "./configs/config.yaml"
# TODO: Fix image paths
# pipeline = Pipeline(load_config(config_path), "", "")


def process_image(image: Image.Image) -> Image.Image:
    """Example image processing function (grayscale conversion)."""
    return image.convert("L")


@app.get("/")
async def home():
    return "The server is running!"


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Read image bytes from the uploaded file
        image_bytes = await file.read()
        print(f"File size: {len(image_bytes)} bytes")

        # Convert the bytes to a PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        print("Image verified successfully!")

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
