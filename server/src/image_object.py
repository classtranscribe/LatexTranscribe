from PIL import Image
import numpy as np
import cv2
import os
import json
import torch


class PrettyFloat(float):
    def __repr__(self):
        return str(round(int(self * 100) / 100.0, 2))


def pretty_floats(obj):
    if isinstance(obj, torch.Tensor):
        return list(map(pretty_floats, obj.tolist()))
    elif isinstance(obj, float):
        return PrettyFloat(obj)
    elif isinstance(obj, dict):
        return dict((k, pretty_floats(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return list(map(pretty_floats, obj))
    elif not isinstance(obj, str):
        PrettyFloat(obj)
    return obj


class ImageObject:
    def __init__(
        self,
        image_path: str | None = None,
        image_name: str | None = None,
        image: Image.Image | None = None,
    ):
        if image_path is not None:
            self.image_path = image_path
            self.image_name = image_path.split("/")[-1].split(".")[0]
            self.image = Image.open(image_path).convert("RGB")
        else:
            self.image_name = image_name
            self.image = image
        self.image_filtered = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        self.image_np = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        self.results = []
        self.visualizations = []
        self.candidates = []

    def get_orig_image(self):
        return self.image

    def get_curr_image(self, as_numpy: bool = False):
        if as_numpy:
            return self.image_filtered
        return Image.fromarray(self.image_filtered)

    def get_candidates(self):
        return self.candidates

    def filter_candidate(self, task, cls, score):
        # priority candidates since models are unique
        if cls == "table":
            return False

        if cls == "formula" and task == "formula_detection":
            return False
        
        if cls == "handwritten":
            return False

        return True

    def apply_bbox(self, box):
        x_min, y_min, x_max, y_max = map(int, box)
        crop = self.image_np[y_min:y_max, x_min:x_max]
        self.image_filtered[y_min:y_max, x_min:x_max] = np.asarray([0, 0, 0])
        return Image.fromarray(crop)

    def create_candidates(self, task, boxes, classes, scores=None, filter_func=None):
        for i in range(len(classes)):
            box = boxes[i]
            cls = classes[i]

            if scores is not None:
                score = scores[i]
            else:
                score = 1

            if filter_func is not None:
                if filter_func(score):
                    continue
            else:
                if self.filter_candidate(task, cls, score):
                    continue

            self.candidates.append((box, cls, self.apply_bbox(box)))

    def add_visualization(self, task, vis):
        self.visualizations.append((task, vis))

    def get_visualizations(self):
        return self.visualizations

    def save_visualizations(self, output_path):
        for i, (task, vis) in enumerate(self.visualizations):
            try:
                cv2.imwrite(
                    os.path.join(output_path, f"{self.image_name}_{task}_{i}.png"), vis
                )
            except:
                try:
                    vis.save(
                        os.path.join(output_path, f"{self.image_name}_{task}_{i}.png")
                    )
                except:
                    pass

    def add_results(self, task, result, cls=None, bbox=None):
        print(f"Adding results for {self.image_name}: {cls}")
        print(result)
        if type(result) == str:
            bbox = [str(round(int(x * 100) / 100.0, 2)) for x in bbox]
            self.results.append(
                pretty_floats({"task": task, "cls": cls, "bbox": bbox, "text": result})
            )

        elif type(result) == list:
            for i in range(len(result)):
                bbox = [str(round(int(x * 100) / 100.0, 2)) for x in result[i]["bbox"]]
                self.results.append(
                    pretty_floats(
                        {
                            "task": task,
                            "cls": result[i]["cls"],
                            "bbox": bbox,
                            "text": result[i]["text"],
                        }
                    )
                )

    def get_results(self):
        return self.results

    def save_results(self, output_path):
        try:
            with open(
                os.path.join(output_path, f"{self.image_name}_results.json"), "w"
            ) as f:
                json.dump(self.results, f)
        except Exception as e:
            print("Failed with", e)
