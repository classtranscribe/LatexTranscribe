# Latex Transcribe

## Quick Start
- To develop locally, [download the ML models](#downloading-the-ml-models-for-pipeline), then [setup the environment without Docker](#local-build-without-docker).
- To build a full Docker image, [download the ML models](#downloading-the-ml-models-for-pipeline), then [build the image (CPU / GPU)](#building-docker-image-locally).


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
- Build and run the Docker image:
    - CPU:
    ```sh
    $ docker build -t latextranscribe -f server/Dockerfile.cpu ./server/
    $ docker run --name latextranscribe-test-server --rm -i -p 8080:80 -t latextranscribe
    ```
    - GPU (CUDA >= 12.4):
    ```sh
    $ docker build -t latextranscribe -f server/Dockerfile.gpu ./server/
    $ docker run --name latextranscribe-test-server --rm --gpus '"device=0"' -i -p 8080:80 -t latextranscribe
    ```
- Go to `http://localhost:8080` and you should now see `Hello, World!`

## Developing Locally
- There are two options to develop locally.
    ### Local build without Docker
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
    - Run the example web server (`uv run` runs `python` in the virtual environment):
        ```sh
        $ uv run uvicorn example-server:app --host 0.0.0.0 --port 8080
        ```
    - Or run the ML pipeline:
        ```sh
        $ uv run main.py
        ```
    ### Docker Compose Watch
    - Run in the project root:
        ```sh
        $ docker compose watch
        ```
    - Go to `http://localhost:8080` and you should now see `Hello, World!`