from src.task_utils.handwritten_utils import fast_base, _remove_padding, set_device_and_dtype
from src.task_utils.utils.preprocessor import PreProcessor
from src.registry import MODEL_REGISTRY
import torch
import torch.nn as nn
from typing import Any
import numpy as np
from src.image_object import ImageObject
from src.data_utils.box_merge import MergeBoxes
import os
from PIL import Image, ImageDraw 

class DetectionPredictor(nn.Module):
    """Implements an object able to localize text elements in a document

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
    """

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: nn.Module,
    ) -> None:
        super().__init__()
        self.pre_processor = pre_processor
        self.model = model.eval()

    @torch.inference_mode()
    def forward(
        self,
        pages: list[np.ndarray | torch.Tensor],
        return_maps: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, np.ndarray]] | tuple[list[dict[str, np.ndarray]], list[np.ndarray]]:
        # Extract parameters from the preprocessor
        preserve_aspect_ratio = self.pre_processor.resize.preserve_aspect_ratio
        symmetric_pad = self.pre_processor.resize.symmetric_pad
        assume_straight_pages = self.model.assume_straight_pages

        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        processed_batches = self.pre_processor(pages)
        _params = next(self.model.parameters())
        self.model, processed_batches = set_device_and_dtype(
            self.model, processed_batches, _params.device, _params.dtype
        )
        predicted_batches = [
            self.model(batch, return_preds=True, return_model_output=True, **kwargs) for batch in processed_batches
        ]
        # Remove padding from loc predictions
        preds = _remove_padding(
            pages,
            [pred for batch in predicted_batches for pred in batch["preds"]],
            preserve_aspect_ratio=preserve_aspect_ratio,
            symmetric_pad=symmetric_pad,
            assume_straight_pages=assume_straight_pages,  # type: ignore[arg-type]
        )

        if return_maps:
            seg_maps = [
                pred.permute(1, 2, 0).detach().cpu().numpy() for batch in predicted_batches for pred in batch["out_map"]
            ]
            return preds, seg_maps
        return preds
    
@MODEL_REGISTRY.register('handwritten_detection_fast')
class HandwrittenDetectionFast:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DetectionPredictor(
            pre_processor=PreProcessor((1024, 1024), 1),
            model=fast_base(pretrained=True)
        )

        self.box_merger = MergeBoxes()
    
    def draw_bounding_boxes(self, image_obj, detection_result, output_folder="images", save_result = False):
        #coordinates = detection_result["results"]["boxes"]
        im = image_obj.get_curr_image(as_numpy=False)  # Get the image as a NumPy array
        
        #im_pil = Image.fromarray(im) 

        draw = ImageDraw.Draw(im)

        for box in detection_result:
            x1, y1, x2, y2 = map(float, box)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        if save_result:
            output_path = os.path.join(output_folder, f"{image_obj.image_name}_with_boxes_test.jpg")

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            im.save(output_path)
            print(f"Image saved at {output_path}")

        return im

    def predict(self, image: list[ImageObject] | ImageObject):
        if type(image) != list:
            image = [image]
        im = [x.get_curr_image(as_numpy=True) for x in image]
        h, w, _ = im[0].shape
        multiplier = np.array([w, h, w, h, 1])
        result = self.model(im)
        boxes = []
        scores = []
        
        for d in result:
            words = d["words"]
            for i in range(words.shape[0]):
                data = (words[i] * multiplier)
                boxes.append(data[:4])
                scores.append(data[4])

        boxes = self.box_merger.merge_boxes(boxes)
        classes = ["handwritten" for _ in range(len(boxes))]
        scores = [1.0 for _ in range(len(boxes))] # TODO: add score

        image_result = self.draw_bounding_boxes(image[0], boxes)


        return {
            "vis": image_result,
            "results": {
                "boxes": boxes,
                "scores": scores,
                "classes": classes,
            }
        }


    
if __name__ == "__main__":
    from src.image_object import ImageObject
    model = HandwrittenDetectionFast(None)
    image = ImageObject(image_path = "/home/nikisun/LatexTranscribe/server/images/h4.jpg")
    out = model.predict(image)
    print(out)