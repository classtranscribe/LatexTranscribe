# Latex Transcribe

## Running the official DockerHub image

(This is sketch and these instructions are untested - assume there are typos, errors, guesses (e.g. Port number), and ommissions that need to be fixed)

docker pull classtranscribe/latextranscribe:latest

docker run -i  -p 8080:8080 -t latextranscribe

## Building Locally

### Local build with Docker

```sh
docker build -t latextranscribe .

docker run -p 127.0.0.1:8000:8000  latextranscribe 
```

Then open http://127.0.0.1:8000 in your favorite browser

### Local build without docker 

(This is sketch and these instructions are untested - assume there are typos, errors, guesses, and ommissions that need to be fixed)


Assuming you have python3.11 installed. Note these instructions essentialy mirror the steps in Dockerfile.

````sh
python3.11 -m venv venv
source ./venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
Hmm todo

````

