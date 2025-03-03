from fastapi import FastAPI, File, UploadFile, status
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from sse_starlette.sse import EventSourceResponse
from PIL import Image
import json
import io
import uuid
import logging
from huey.contrib.asyncio import aget_result

from src.pipeline_task import run_pipeline

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger = logging.getLogger("uvicorn.error")

tasks = {}
task_images = {}


def concat_visualizations(visualizations):
    # Convert all visualizations to PIL Images if they aren't already
    pil_images = []
    for _, vis in visualizations:
        if isinstance(vis, Image.Image):
            pil_images.append(vis)
        else:
            pil_images.append(Image.fromarray(vis))

    # Calculate total height and max width
    total_height = sum(img.height for img in pil_images)
    max_width = max(img.width for img in pil_images)

    # Create new image with combined height
    combined_image = Image.new("RGB", (max_width, total_height))

    # Paste images vertically
    y_offset = 0
    for img in pil_images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height
    return combined_image


@app.get("/")
async def home():
    return "The backend is running!"


async def task_update_generator(task_id: uuid.UUID):
    global tasks
    if task_id not in tasks:
        yield {"event": "error", "data": "Task not found"}
        return
    visualizations, results = await aget_result(tasks[task_id])

    task_images[task_id] = concat_visualizations(visualizations)
    logger.debug(f"{task_images[task_id]}")
    logger.debug(f"Task {task_id} finished")
    yield {"event": "result", "data": json.dumps(results)}
    yield {"event": "close", "data": "Connection closed"}
    del tasks[task_id]
    return


@app.get(
    "/task/{task_id}",
    summary="SSE endpoint to get task updates",
    response_description="Streaming task updates",
)
async def message_stream(task_id: uuid.UUID):
    return EventSourceResponse(task_update_generator(task_id))


@app.get(
    "/task/{task_id}/image",
    summary="Get the processed image for a task",
    response_description="Processed image",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def get_task_image(task_id: uuid.UUID):
    if task_id not in task_images:
        return Response(status_code=status.HTTP_404_NOT_FOUND)
    # Convert numpy array to PIL Image and then to PNG bytes
    img_bytes = io.BytesIO()
    # Image.fromarray(task_images[task_id]).save(img_bytes, format="PNG")
    task_images[task_id].save(img_bytes, format="PNG")
    return Response(content=img_bytes.getvalue(), media_type="image/png")


@app.post(
    "/upload",
    summary="Upload an image for processing",
    response_description="Sucess message and corresponding task ID",
    status_code=status.HTTP_201_CREATED,
)
async def upload_image(file: UploadFile = File(...)):
    try:
        # Read image bytes from the uploaded file
        image_bytes = await file.read()
        # Convert the bytes to a PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        task_id = uuid.uuid4()
        logger.debug(
            f"Created task (task_id: {task_id}, filename: {file.filename}, size: {len(image_bytes)} bytes)"
        )
        tasks[task_id] = run_pipeline(name=file.filename, image=image)
        return {"message": "Success", "task_id": str(task_id)}
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return JSONResponse(
            content={"message": "Failure"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
