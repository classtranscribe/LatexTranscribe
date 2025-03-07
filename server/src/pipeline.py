from PIL import Image
from src.image_object import ImageObject
from src.utils import get_image_paths
from src.registry import MODEL_REGISTRY
from src.utils import get_accelerator

# for registry purposes
from src.tasks import (
    formula_detection,
    formula_recognition,
    omniparser,
    layout_detection,
    table_recognition,
)


class Pipeline:
    def __init__(
        self, config: str, input_path: str | None = None, output_path: str | None = None
    ):
        self.models = {}
        for task in config["models"]:
            ModelClass = MODEL_REGISTRY.get(config["models"][task]["model_name"])
            self.models[task] = ModelClass(config["models"][task])

        self.images = {}

        if input_path is not None:
            image_paths = get_image_paths(input_path)
            for image_path in image_paths:
                image_name = image_path.split("/")[-1].split(".")[0]
                self.images[image_name] = ImageObject(image_path)
                print("Loaded", image_name)

        self.output_path = output_path

    def add_image(self, name, image: Image.Image):
        self.images[name] = ImageObject(image=image, image_name=name)

    def detect_candidates(self, task, images: dict[str, ImageObject] | None = None):
        if images is None:
            images = self.images

        for image_name, image in images.items():
            print(f"Detecting {task} for {image_name}")
            out = self.models[task].predict(image.get_curr_image())
            print(out)
            print("-" * 50)
            image.add_visualization(task, out["vis"])
            image.create_candidates(
                task,
                out["results"]["boxes"],
                out["results"]["classes"],
                out["results"]["scores"],
            )

    def transcribe_image(self, images: dict[str, ImageObject] | None = None):
        if images is None:
            images = self.images
        for image_name, image in images.items():
            print(f"Transcribing {image_name}")
            candidates = image.get_candidates()
            print(candidates)
            for box, cls, crop in candidates:
                # should all be one of these two for now
                if cls in ["formula", "table"]:
                    task = f"{cls}_recognition"
                else:
                    continue

                out = self.models[task].predict(crop)
                print(out)
                if out["vis"] is not None:
                    image.add_visualization(task, out["vis"])
                image.add_results(task, out["results"], cls, box)

            out = self.models["base_recognition"].predict(image.get_curr_image())
            print(out)
            if out["vis"] is not None:
                image.add_visualization("base_recognition", out["vis"])
            image.add_results("base_recognition", out["results"])
            print("-" * 50)

    # stateless
    def predict_image(self, name: str, image: Image.Image) -> tuple[list, list]:
        print(
            f"Predicting using device={get_accelerator(no_mps=False)} (backup=cpu)..."
        )
        image_obj = ImageObject(image=image, image_name=name)
        image_iter = {name: image_obj}
        self.detect_candidates("layout_detection", images=image_iter)
        self.detect_candidates("formula_detection", images=image_iter)
        self.transcribe_image(images=image_iter)
        image_obj.save_visualizations(".")
        return image_obj.get_visualizations(), image_obj.get_results()

    def predict(self, save=False):
        print(
            f"Predicting using device={get_accelerator(no_mps=False)} (backup=cpu)..."
        )
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
