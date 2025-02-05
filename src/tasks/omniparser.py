from src.utils import get_som_labeled_img, check_ocr_box_image, get_yolo_model
import torch
from ultralytics import YOLO
from PIL import Image
from typing import Dict, Tuple, List
import io
import base64
import json
from src.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("omniparser")
class Omniparser(object):
    def __init__(self, config):
        self.config = config
        
        self.som_model = get_yolo_model(model_path=config['model_path'])

    def predict(self, image):
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box_image(image, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9})
        text, ocr_bbox = ocr_bbox_rslt

        if ocr_bbox is None:
            return {
                'vis': None,
                'results': None
            }

        draw_bbox_config = self.config['draw_bbox_config']
        BOX_TRESHOLD = self.config['box_threshold']
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image, self.som_model, BOX_TRESHOLD = BOX_TRESHOLD, output_coord_in_ratio=False, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=None, ocr_text=text,use_local_semantics=False)

        dino_image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
        # formating output
        return_list = [{'bbox': [coord[0], coord[1], coord[0] + coord[2], coord[1] + coord[3]],
                        'text': parsed_content_list[i].split(': ')[1], 'cls':'text'} for i, (k, coord) in enumerate(label_coordinates.items()) if i < len(parsed_content_list)]
        return_list.extend(
            [{'bbox': [coord[0], coord[1], coord[0] + coord[2], coord[1] + coord[3]],
                        'text': 'None', 'cls':'icon'} for i, (k, coord) in enumerate(label_coordinates.items()) if i >= len(parsed_content_list)]
              )
        
        return {
            'vis': dino_image,
            'results': return_list
        }
