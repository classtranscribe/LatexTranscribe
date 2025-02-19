from src.image_object import ImageObject
from src.utils import get_image_paths
from src.registry import MODEL_REGISTRY

# for registry purposes
from src.tasks import (
    formula_detection,
    formula_recognition,
    layout_detection,
    table_recognition,
)


class Pipeline:
    def __init__(self, config, input_path: str, outut_path: str):
        self.models = {}
        for task in config["models"]:
            ModelClass = MODEL_REGISTRY.get(config["models"][task]["model_name"])
            self.models[task] = ModelClass(config["models"][task])

        self.images = {}

        image_paths = get_image_paths(input_path)
        for image_path in image_paths:
            image_name = image_path.split("/")[-1].split(".")[0]
            self.images[image_name] = ImageObject(image_path)

        self.output_path = outut_path

    def add_image(self, name, image):
        self.images[name] = ImageObject(image=image, image_name=name)

    def detect_candidates(self, task):
        for image_name in self.images:
            out = self.models[task].predict(self.images[image_name].get_curr_image())
            print(out["results"])
            self.images[image_name].add_visualization(task, out["vis"])
            self.images[image_name].create_candidates(
                task,
                out["results"]["boxes"],
                out["results"]["classes"],
                out["results"]["scores"],
            )

    def transcribe_image(self):
        for image_name in self.images:
            image = self.images[image_name]
            candidates = image.get_candidates()
            for box, cls, crop in candidates:
                # should all be one of these two for now
                if cls in ["formula", "table"]:
                    task = f"{cls}_recognition"
                else:
                    continue

                out = self.models[task].predict(crop)
                if out["vis"] is not None:
                    image.add_visualization(task, out["vis"])
                image.add_results(task, out["results"], cls, box)

    def predict(self, save=False):
        self.detect_candidates("layout_detection")
        self.detect_candidates("formula_detection")
        self.transcribe_image()

        if save:
            for image_name in self.images:
                print(image_name)
                print(self.images[image_name].results)
                print("")
                self.images[image_name].save_visualizations(self.output_path)
                self.images[image_name].save_results(self.output_path)
        else:
            return self.images
