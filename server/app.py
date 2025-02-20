import os
from flask import Flask, jsonify, flash, request, redirect, url_for, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import json
from src.pipeline import Pipeline

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png"}
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CORS(app)
if not os.path.exists(UPLOAD_FOLDER):
    print("no upload", flush=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

pipeline = Pipeline("./app_config.yaml")

def process_image(image: Image.Image) -> Image.Image:
    """Example image processing function (grayscale conversion)."""
    vis, res = pipeline.predict_image("current", image)
    return vis, res


@app.route("/upload", methods=["POST"])
def upload_image():
    print(request.files, flush=True)
    if "file" not in request.files:
        return jsonify({"message": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "No file uploaded"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image = Image.open(file.stream)
        visualizations, transcriptions = process_image(image)
        for visualization in visualizations[1:]:
            task, img = visualization
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{task}_{filename}")
            img.save(filepath, format="PNG")

        with open(os.path.join(app.config["UPLOAD_FOLDER"], 
                               f"results.json"), "w") as f:
                json.dump(transcriptions, f)

        processed_image = visualizations[0][1] # layout detection output
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        processed_image.save(filepath, format="PNG")
        
        return send_file(filepath, mimetype = "image/png")

    return jsonify({"message": "Invalid file type. Only .png is allowed"}), 400


@app.route("/")
def hello_world():
    return jsonify({"message": "Hello, World!"})


@app.route("/sanitycheck")
def vue_setup():
    return jsonify({"message": "Server setup!"})
