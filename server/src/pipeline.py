from src.image_object import ImageObject
from src.utils import get_image_paths
from src.registry import MODEL_REGISTRY
from PIL import Image

# for registry purposes
from src.tasks import (
    formula_detection,
    formula_recognition,
    layout_detection,
    table_recognition,
)


class Pipeline:
    def __init__(self, config: str, input_path: str, output_path: str):
        self.models = {}
        for task in config["models"]:
            ModelClass = MODEL_REGISTRY.get(config["models"][task]["model_name"])
            self.models[task] = ModelClass(config["models"][task])

        self.images = {}

        image_paths = get_image_paths(input_path)
        for image_path in image_paths:
            image_name = image_path.split("/")[-1].split(".")[0]
            self.images[image_name] = ImageObject(image_path)

        self.output_path = output_path

    def add_image(self, name, image: Image.Image):
        self.images[name] = ImageObject(image=image, image_name=name)

    def detect_candidates(self, task, images=None):
        if images is None:
            images = self.images

        for image_name in images:
            out = self.models[task].predict(images[image_name].get_curr_image())
            print(out["results"])
            images[image_name].add_visualization(task, out["vis"])
            images[image_name].create_candidates(
                task,
                out["results"]["boxes"],
                out["results"]["classes"],
                out["results"]["scores"],
            )

    def transcribe_image(self, images=None):
        if images is None:
            images = self.images

        for image_name in images:
            image = images[image_name]
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

    # stateless
    def predict_image(self, name: str, image: Image.Image):
        image_obj = ImageObject(image=image, image_name=name)
        image_iter = {name: image_obj}
        self.detect_candidates("layout_detection", images=image_iter)
        self.detect_candidates("formula_detection")
        self.transcribe_image()


        return image_obj.get_visualizations(), image_obj.get_results()

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
