# Latex Transcribe

## Running the official DockerHub image

(This is sketch and these instructions are untested - assume there are typos, errors, guesses (e.g. Port number), and ommissions that need to be fixed)

```sh
$ docker pull classtranscribe/latextranscribe:latest
$ docker run -i -p 8080:8080 -t latextranscribe
```

## Building Locally
- Recommended: Install the [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) package manager.

#### Downloading the ML models for Pipeline
- Download the ML models first (warning: large download).
    - With `uv` (no dependencies required thanks to [inline script dependencies](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies)):
    ```sh
    $ cd server && uv run download_models.py
    ```
    - Or with `pip`:
    ```sh
    $ pip install huggingface-hub
    $ cd server && python download_models.py
    ```
Now, there are two options to build and run the project.

#### Local build with Docker
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

#### Local build without Docker
- Work in the `server` directory (`$ cd server/`).
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

## Developing Locally
- Either do [Local build without Docker](#local-build-without-docker), or use `docker compose` in the project root:
    ```sh
    $ docker compose watch
    ```
- Go to `http://localhost:8080` and you should now see `Hello, World!`
