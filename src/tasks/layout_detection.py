import os
import cv2
import numpy as np
from PIL import Image

from src.registry import MODEL_REGISTRY
from .layoutlmv3_util.model_init import Layoutlmv3_Predictor
from src.utils import visualize_bbox

@MODEL_REGISTRY.register("layout_detection_layoutlmv3")
class LayoutDetectionLayoutlmv3:
    def __init__(self, config):
        """
        Initialize the LayoutDetectionYOLO class.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        # Mapping from class IDs to class names
        self.id_to_names = {
            0: 'title', 
            1: 'text',
            2: 'abandon', 
            3: 'figure', 
            4: 'figure_caption', 
            5: 'table', 
            6: 'table_caption', 
            7: 'table_footnote', 
            8: 'formula', 
            9: 'formula_caption'
        }
        self.model = Layoutlmv3_Predictor(config.get('model_path', None))
        self.visualize = config.get('visualize', False)

    def predict(self, image):
        """
        Predict layouts in images.

        Args:
            images (list): List of images to be predicted.
            result_path (str): Path to save the prediction results.
            image_ids (list, optional): List of image IDs corresponding to the images.

        Returns:
            list: List of prediction results.
        """
        layout_res = self.model(np.array(image), ignore_catids=[])
        poly = np.array([det["poly"] for det in layout_res["layout_dets"]])
        boxes = poly[:, [0,1,4,5]] 
        scores = np.array([det["score"] for det in layout_res["layout_dets"]])
        classes = np.array([det["category_id"] for det in layout_res["layout_dets"]])
        
        vis_result = visualize_bbox(image, boxes, classes, scores, self.id_to_names)


        return {
            "vis": vis_result,
            "results": {
                "boxes": boxes.tolist(),
                "scores": scores,
                "classes": [self.id_to_names[classes[i]] for i in range(len(classes))],
            }
        }
