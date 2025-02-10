# Latex Transcribe

## Running the official DockerHub image

(This is sketch and these instructions are untested - assume there are typos, errors, guesses (e.g. Port number), and ommissions that need to be fixed)

```sh
$ docker pull classtranscribe/latextranscribe:latest
$ docker run -i -p 8080:8080 -t latextranscribe
```

## Building Locally

### Local build with Docker
- Currently only installs CPU dependencies.
```sh
$ docker build -t latextranscribe -f Dockerfile.cpu .
$ docker run --name latextranscribe-test-server --rm -i -p 8080:80 -t latextranscribe
```


### Local build without Docker

- Install the [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) package manager.
- Create a virtual environment and install the dependencies:
````sh
$ uv sync --extra cpu
````
- Run the example web server (`uv run` runs `python` in the virtual environment):
````sh
$ uv run uvicorn example-server:app --host 0.0.0.0 --port 8080
````
- Run the pipeline:
````sh
$ uv run main.py
````

