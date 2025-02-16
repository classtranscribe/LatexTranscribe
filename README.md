# Latex Transcribe

## Quick Start
### Backend
- To develop locally, [download the ML models](#downloading-the-ml-models-for-pipeline), then [setup the backend/frontend environment without Docker](#local-build-without-docker).
- To build a full Docker image, [download the ML models](#downloading-the-ml-models-for-pipeline), then [build the backend (CPU / GPU) and frontend images](#building-docker-image-locally).

## Running the official DockerHub image
- Only up-to-date with the main branch.
```sh
$ docker pull classtranscribe/latextranscribe:latest
$ docker run -i -p 8080:80 -t latextranscribe
```

## Downloading the ML models for Pipeline
- Recommended: Install the [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) package manager.
- First, download the ML models (warning: large download).
    - With `uv` (no dependencies required thanks to [inline script dependencies](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies)):
    ```sh
    $ cd server && uv run download_models.py
    ```
    - Or with `pip`:
    ```sh
    $ pip install huggingface-hub
    $ cd server && python download_models.py
    ```
## Building Docker Image Locally
### Backend
- Build and run the backend Docker image:
    - CPU:
    ```sh
    $ docker build -t latextranscribe-backend -f server/Dockerfile.cpu ./server/
    $ docker run --name latextranscribe-backend --rm -i -p 8081:80 -t latextranscribe-backend
    ```
    - GPU (CUDA >= 12.4):
    ```sh
    $ docker build -t latextranscribe-backend -f server/Dockerfile.gpu ./server/
    $ docker run --name latextranscribe-backend --rm --gpus '"device=0"' -i -p 8081:80 -t latextranscribe-backend
    ```
- Go to `http://localhost:8081` and you should now see `The server is running!`
### Frontend
- Build and run the frontend Docker image:
    ```sh
    $ docker build -t latextranscribe-frontend -f frontend/Dockerfile ./frontend/
    $ docker run --name latextranscribe-frontend --rm -i -p 8080:80 -t latextranscribe-frontend
    ```
- Go to `http://localhost:8080` and you should now see the frontend.

## Developing Locally
- There are two options to develop locally.
    ### Docker Compose Watch
    - Run in the project root:
        ```sh
        $ docker compose up --build --watch
        ```
    - Go to `http://localhost:8080` and you should now see the frontend.
    ### Local build without Docker
    #### Backend
    - Switch into the `server` directory:
        ```sh
        $ cd server/
        ```
    - Create a virtual environment and install the dependencies:
        - CPU:
        ```sh
        $ uv sync --extra cpu
        ```
        - GPU (CUDA >= 12.4):
        ```sh
        $ uv sync --extra cu124
        ```
    - Run the server (`uv run` runs `python` in the virtual environment):
        ```sh
        $ uv run uvicorn app:app --host 0.0.0.0 --port 8081
        ```
    - TESTING – Or run the ML pipeline:
        ```sh
        $ uv run main.py
        ```
    #### Frontend
    - Switch into the `frontend` directory:
        ```sh
        $ cd frontend/
        ```
    - Install the dependencies:
        ```sh
        $ npm install
        ```
    - Run the development server:
        ```sh
        $ npm run dev
        ```
    - Go to `http://localhost:5173` and you should now see the frontend.
