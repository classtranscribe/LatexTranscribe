import os
from flask import Flask, jsonify, flash, request, redirect, url_for, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png"}
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CORS(app)
if not os.path.exists(UPLOAD_FOLDER):
    print("no upload", flush=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(image: Image.Image) -> Image.Image:
    """Example image processing function (grayscale conversion)."""
    return image.convert("L")


@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"message": "No file part"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "No file uploaded"})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image = Image.open(file.stream)
        processed_image = process_image(image)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        processed_image.save(filepath, format="PNG")
        return jsonify(
            {"message": "Image uploaded successfully", "file_path": filepath}
        )

    return jsonify({"message": "Invalid file type. Only .png is allowed"})


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/sanitycheck")
def vue_setup():
    return jsonify({"message": "Server setup!"})
