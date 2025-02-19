# Latex Transcribe

## Quick Start
### Backend
- To develop locally, [download the ML models](#downloading-the-ml-models-for-pipeline), then [setup the backend/frontend environment without Docker](#local-build-without-docker).
- To build a full Docker image, [download the ML models](#downloading-the-ml-models-for-pipeline), then [build the backend (CPU / GPU) and frontend images](#building-docker-image-locally).

## Running the official DockerHub image
- Only up-to-date with the main branch.
```sh
docker pull classtranscribe/latextranscribe:latest
docker run -i -p 8080:80 -t latextranscribe
```

## Downloading the ML models for Pipeline
- Recommended: Install the [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) package manager.
- First, download the ML models (warning: large download).
    - With `uv` (no dependencies required thanks to [inline script dependencies](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies)):
    ```sh
    cd server && uv run download_models.py
    ```
    - Or with `pip`:
    ```sh
    pip install huggingface-hub
    cd server && python download_models.py
    ```
## Building Docker Image Locally
### Backend
- Build and run the backend Docker image:
    - CPU:
    ```sh
    docker build -t latextranscribe -f server/Dockerfile.cpu ./server/
    docker run --name latextranscribe --rm -i -p 8081:80 -t latextranscribe
    ```
    - GPU (CUDA >= 12.4):
    ```sh
    docker build -t latextranscribe -f server/Dockerfile.gpu ./server/
    docker run --name latextranscribe --rm --gpus '"device=0"' -i -p 8081:80 -t latextranscribe
    ```
- Go to `http://localhost:8081` and you should now see `The server is running!`
### Frontend
- Build and run the frontend Docker image:
    ```sh
    docker build -t latextranscribe-frontend -f frontend/Dockerfile ./frontend/
    docker run --name latextranscribe-frontend --rm -i -p 8080:80 -t latextranscribe-frontend
    ```
- Go to `http://localhost:8080` and you should now see the frontend.

## Developing Locally
- There are two options to develop locally.
    ### Docker Compose Watch
    - Run in the project root:
        ```sh
        docker compose up --build --watch
        ```
    - Go to `http://localhost:8080` and you should now see the frontend.
    ### Local build without Docker
    #### Backend
    - Switch into the `server` directory:
        ```sh
        cd server/
        ```
    - Create a virtual environment and install the dependencies:
        - CPU:
        ```sh
        uv venv --seed # creates the virtual environment
        uv sync --extra cpu --no-install-package detectron2 --no-install-package struct-eqtable # installs dependencies other than the two source packages
        uv sync --extra cpu # installs the two source packages
        ```
        - GPU (CUDA >= 12.4):
        ```sh
        uv venv --seed # creates the virtual environment
        uv sync --extra cu124 --no-install-package detectron2 --no-install-package struct-eqtable # installs dependencies other than the two source packages
        uv sync --extra cu124 # installs the two source packages
        ```
    - Run the server (`uv run` runs `python` in the virtual environment):
        ```sh
        uv run gunicorn -w 1 'app:app' --bind 0.0.0.0:8081 --reload --access-logfile '-' --log-level debug
        ```
    - TESTING – Or run the ML pipeline:
        ```sh
        uv run main.py
        ```
    #### Frontend
    - Switch into the `frontend` directory:
        ```sh
        cd frontend/
        ```
    - Install the dependencies:
        ```sh
        npm install
        ```
    - Run the development server:
        ```sh
        npm run dev
        ```
    - Go to `http://localhost:5173` and you should now see the frontend.

## Others
- To add a project dependency: [`uv add`](https://docs.astral.sh/uv/reference/cli/#uv-add)
- To sync the project dependencies again: `uv sync --extra cpu` or `uv sync --extra cu124`