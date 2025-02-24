from PIL import Image
import yaml
from huey import SqliteHuey


CONFIG_PATH = "./configs/config.yaml"

huey = SqliteHuey(filename="./tasks.db")
pipeline = None


@huey.on_startup()
def setup_pipeline():
    # Import here to avoid backend also importing the pipeline
    from src.pipeline import Pipeline

    global pipeline
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    pipeline = Pipeline(config)
    print("Pipeline setup complete.")


@huey.task()
def run_pipeline(name: str, image: Image.Image):
    """Run the pipeline on a single image and return the results."""
    print(f"Running pipeline {name} [{image.size}]")
    global pipeline
    return pipeline.predict_image(name, image)
