import torch

from PIL import Image
from struct_eqtable import build_model
from src.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("table_parsing_struct_eqtable")
class TableParsingStructEqTable:
    def __init__(self, config):
        """
        Initialize the TableParsingStructEqTable class.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        assert torch.cuda.is_available(), "CUDA must be available for StructEqTable model."

        self.model_dir = config.get('model_path', 'U4R/StructTable-InternVL2-1B')
        self.max_new_tokens = config.get('max_new_tokens', 1024)
        self.max_time = config.get('max_time', 30)

        self.lmdeploy = config.get('lmdeploy', False)
        self.flash_attn = config.get('flash_attn', True)
        self.batch_size = config.get('batch_size', 1)
        self.default_format = config.get('output_format', 'latex')

        # Load the StructEqTable model
        self.model = build_model(
            model_ckpt=self.model_dir,
            max_new_tokens=self.max_new_tokens,
            max_time=self.max_time,
            lmdeploy=self.lmdeploy,
            flash_attn=self.flash_attn,
            batch_size=self.batch_size,
        ).cuda()

    def predict(self, image):

        results = self.model(
            image, output_format=self.default_format
        )


        return {
            'vis': None,
            'results': results[0]
        }
