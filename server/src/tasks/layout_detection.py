import os
import numpy as np
import cv2
import torch
from src.registry import MODEL_REGISTRY
from src.utils import visualize_bbox

@MODEL_REGISTRY.register('layout_detection_yolo')
class LayoutDetectionYOLO:
    def __init__(self, config):
        """
        Initialize the LayoutDetectionYOLO class.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        # Mapping from class IDs to class names
        self.id_to_names = {
            0: 'title',
            1: 'plain text',
            2: 'abandon',
            3: 'figure',
            4: 'figure_caption',
            5: 'table',
            6: 'table_caption',
            7: 'table_footnote',
            8: 'isolate_formula',
            9: 'formula_caption'
        }

        # Load the YOLO model from the specified path
        try:
            from doclayout_yolo import YOLOv10
            self.model = YOLOv10(config['model_path'])
        except AttributeError:
            from ultralytics import YOLO
            self.model = YOLO(config['model_path'])

        # Set model parameters
        self.img_size = config.get('img_size', 1280)
        self.conf_thres = config.get('conf_thres', 0.25)
        self.iou_thres = config.get('iou_thres', 0.45)
        self.visualize = config.get('visualize', False)
        self.nc = config.get('nc', 10)
        self.workers = config.get('workers', 8)
        self.device = config.get('device', 'cpu')

        if self.iou_thres > 0:
            import torchvision
            self.nms_func = torchvision.ops.nms

    def predict(self, image):
        result = self.model.predict(image, imgsz=self.img_size, conf=self.conf_thres, iou=self.iou_thres, verbose=False, device=self.device)[0]
        if self.visualize:
            boxes = result.__dict__['boxes'].xyxy
            classes = result.__dict__['boxes'].cls
            scores = result.__dict__['boxes'].conf

            if self.iou_thres > 0:
                indices = self.nms_func(boxes=torch.Tensor(boxes), scores=torch.Tensor(scores),iou_threshold=self.iou_thres)
                boxes, scores, classes = boxes[indices], scores[indices], classes[indices]
                if len(boxes.shape) == 1:
                    boxes = np.expand_dims(boxes, 0)
                    scores = np.expand_dims(scores, 0)
                    classes = np.expand_dims(classes, 0)

            vis_result = visualize_bbox(image, boxes, classes, scores, self.id_to_names)

        print(boxes, classes, scores)
        classes = classes.tolist()
        return {
            "vis": vis_result,
            "results": {
                "boxes": boxes.tolist(),
                "scores": scores.tolist(),
                "classes": [self.id_to_names[classes[i]] for i in range(len(classes))],
            },
        }
