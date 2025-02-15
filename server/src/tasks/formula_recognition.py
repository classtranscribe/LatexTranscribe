import os
import logging
import argparse

import cv2
import torch
import numpy as np
from PIL import Image
import unimernet.tasks as tasks
from unimernet.common.config import Config
from unimernet.processors import load_processor
import os

from src.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register('formula_recognition_unimernet')
class FormulaRecognitionUniMERNet:
    def __init__(self, config):
        """
        Initialize the FormulaRecognitionUniMERNet class.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = config['model_path']
        self.cfg_path = config.get('cfg_path', "pdf_extract_kit/configs/unimernet.yaml")
        self.batch_size = config.get('batch_size', 1)

        # Load the UniMERNet model
        self.model, self.vis_processor = self.load_model_and_processor()

    def load_model_and_processor(self):
        try:
            args = argparse.Namespace(cfg_path=self.cfg_path, options=None)
            cfg = Config(args)
            cfg.config.model.pretrained = os.path.join(self.model_dir, "pytorch_model.pth")
            cfg.config.model.model_config.model_name = self.model_dir
            cfg.config.model.tokenizer_config.path = self.model_dir
            task = tasks.setup_task(cfg)
            model = task.build_model(cfg).to(self.device)
            vis_processor = load_processor('formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)
            return model, vis_processor
        except Exception as e:
            logging.error(f"Error loading model and processor: {e}")
            raise
    
    def predict(self, image):
        # Process the image using the visual processor and prepare it for the model
        image = self.vis_processor(image).unsqueeze(0).to(self.device)
        # Generate the prediction using the model
        output = self.model.generate({"image": image})
        pred = output["pred_str"][0]
    
        return {'vis': None, 'results': pred}