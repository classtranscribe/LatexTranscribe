import os
import cv2
import torch
from ultralytics import YOLO
from src.registry import MODEL_REGISTRY
from src.utils import visualize_bbox
import numpy as np
from PIL import Image
from collections import defaultdict



@MODEL_REGISTRY.register('formula_detection_yolo')
class FormulaDetectionYOLO:
    def __init__(self, config):
        """
        Initialize the FormulaDetectionYOLO class.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        # Mapping from class IDs to class names
        self.id_to_names = {
            0: 'inline',
            1: 'isolated'
        }

        # Load the YOLO model from the specified path
        self.model = YOLO(config['model_path'])

        # Set model parameters
        self.img_size = config.get('img_size', 1280)
        self.pdf_dpi = config.get('pdf_dpi', 200)
        self.conf_thres = config.get('conf_thres', 0.25)
        self.iou_thres = config.get('iou_thres', 0.45)
        self.visualize = config.get('visualize', True)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = config.get('batch_size', 1)

    def predict(self, image):
        """
        Predict formulas in images.

        Args:
            images (list): List of images to be predicted.
            result_path (str): Path to save the prediction results.
            image_ids (list, optional): List of image IDs corresponding to the images.

        Returns:
            list: List of prediction results.
        """
        result = self.model.predict(image, imgsz=self.img_size, conf=self.conf_thres, iou=self.iou_thres, verbose=False)[0]
        if self.visualize:
            boxes = result.__dict__['boxes'].xyxy
            classes = result.__dict__['boxes'].cls
            scores = result.__dict__['boxes'].conf
            
            vis_result = visualize_bbox(image, boxes, classes, scores, self.id_to_names)

        return {
            'vis': vis_result,
            'results': {
                "boxes": boxes.tolist(),
                "scores": scores,
                "classes": ["formula" for i in range(len(classes))],
            }
        }