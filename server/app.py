from fastapi import FastAPI, File, UploadFile, status, APIRouter
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
router = APIRouter(prefix="/api")
logger = logging.getLogger("uvicorn.error")

tasks = {}
task_images = {}


@app.get("/")
async def home():
    return "The backend is running!"


async def task_update_generator(task_id: uuid.UUID):
    global tasks
    if task_id not in tasks:
        yield {"event": "error", "data": "Task not found"}
        return
    visualizations, results = await aget_result(tasks[task_id])
    task_images[task_id] = visualizations[0][1]
    logger.debug(f"{task_images[task_id]}")
    logger.debug(f"Task {task_id} finished")
    yield {"event": "result", "data": json.dumps(results)}
    yield {"event": "close", "data": "Connection closed"}
    del tasks[task_id]
    return


@router.get(
    "/task/{task_id}",
    summary="SSE endpoint to get task updates",
    response_description="Streaming task updates",
)
async def message_stream(task_id: uuid.UUID):
    return EventSourceResponse(task_update_generator(task_id))


@router.get(
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
    Image.fromarray(task_images[task_id]).save(img_bytes, format="PNG")
    return Response(content=img_bytes.getvalue(), media_type="image/png")


@router.post(
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


app.include_router(router)
