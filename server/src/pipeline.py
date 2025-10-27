import logging
import tempfile
import json
import io
from pathlib import Path
from PIL import Image
from collections import defaultdict
from typing import Any, Dict, List, Tuple 

try:
    from mineru.backend.pipeline.pipeline_analyze import doc_analyze
    from mineru.data.data_reader_writer.filebase import FileBasedDataWriter

except ImportError as e:
    print(f"Error: 'mineru' package components not found: {e}")
    doc_analyze = None


logger = logging.getLogger("uvicorn.error")

# Helper functions to convert the MinerU output into bbox for frontend

def _bbox_to_xywh(b: Any) -> Dict[str, float]:
    if isinstance(b, (list, tuple)) and len(b) == 4:
        x1, y1, x2, y2 = map(float, b)
        return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
    return {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}

def normalize_for_taxonomy(raw_mineru_output: Any) -> Dict[str, Any]:
    logger.info("Starting Normalization")
    content_list = None
    
    if isinstance(raw_mineru_output, tuple):
        logger.info(f"MinerU output is a tuple with {len(raw_mineru_output)} elements.")
        
        for i, item in enumerate(raw_mineru_output):
            logger.info(f"Tuple item {i} type: {type(item)}")

            if isinstance(item, list):
                logger.info(f"  > Found a list at index {i} with {len(item)} elements.")
                
                if content_list is None:
                    logger.info(f"  > ASSIGNING content_list from index {i}")
                    content_list = item
                    
                    if len(item) > 0:
                        logger.info(f"  > First element of list: {str(item[0])[:200]}...")

        if content_list is None:
            logger.error(f"MinerU output was a tuple, but NO LIST was found inside.")
            return {"pages": []}
            
    elif isinstance(raw_mineru_output, list):
        logger.info("MinerU output was a list (original expectation).")
        content_list = raw_mineru_output
        
    else:
        logger.error(f"MinerU output was not a list or tuple, returning empty. Got: {type(raw_mineru_output)}")
        return {"pages": []}
    
    if not content_list: 
        logger.warning("MinerU content list is empty (this is expected if using 'ch' on 'en' text).")
        return {"pages": []}
        
    pages = defaultdict(list)
    for el in content_list:
        if not isinstance(el, dict) or "type" not in el or "bbox" not in el:
            logger.warning(f"Skipping invalid element in content_list: {el}")
            continue
        pidx = int(el.get("page_idx", 0))
        bbox = _bbox_to_xywh(el["bbox"])
        t = el["type"]
        attrs = {}
        if "text_level" in el: attrs["text_level"] = el["text_level"]
        if t == "image":
            if "img_path" in el: attrs["img_path"] = el["img_path"]
            if "image_caption" in el: attrs["image_caption"] = el["image_caption"]
            if "image_footnote" in el: attrs["image_footnote"] = el["image_footnote"]
        item = {
            "type": "figure" if t == "image" else t,
            "bbox": bbox,
            "confidence": 1.0,  
            "text": el.get("text"),
            "latex": el.get("latex"),
            "html": el.get("html"),
            "attributes": attrs,
        }
        pages[pidx].append(item)
    
    page_count = len(pages)
    element_count = sum(len(elems) for elems in pages.values())
    logger.info(f"Normalization complete: {page_count} pages, {element_count} total elements.")
    
    result = {"pages": [{"page_index": i, "elements": elems} for i, elems in sorted(pages.items())]}
    return result

class Pipeline:
    def __init__(self, config: str):
        if doc_analyze is None:
            logger.error("MinerU 2.0 is not installed. Pipeline cannot be created.")
            self.initialized = False
            return
        logger.info("Initializing MinerU 2.0 Pipeline...")
        self.initialized = True
        logger.info("MinerU 2.0 Pipeline initialized successfully.")

    def predict_image(self, name: str, image: Image.Image) -> tuple[list, dict]:
        if not self.initialized:
            logger.error("MinerU model is not initialized. Cannot predict.")
            return [], {"pages": []}

        logger.info(f"Running MinerU 2.0 prediction for {name}...")

        try:
            pdf_buffer = io.BytesIO()
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            image.save(pdf_buffer, 'PDF', resolution=100.0)
            pdf_bytes = pdf_buffer.getvalue()
            logger.info(f"Converted image to in-memory PDF bytes")
            
            # Call the MinerU model
            raw_result_data = doc_analyze(
                pdf_bytes_list=[pdf_bytes],
                lang_list=['en'], 
                parse_method="auto"
            )
            
            logger.info(f"MinerU pipeline raw output type: {type(raw_result_data)}")
            
            normalized_results = normalize_for_taxonomy(raw_result_data)
            
            return [], normalized_results
        
        except Exception as e:
            logger.error(f"Error during MinerU prediction for {name}: {e}")
            logger.exception(e)
            return [], {"pages": []} 
    
    def cleanup(self):
        logger.info("Pipeline cleanup (no-op).")
        pass
    
    def __del__(self):
        if hasattr(self, 'initialized') and self.initialized:
            self.cleanup()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = Pipeline(config={})
    try:
        test_image = Image.new('RGB', (800, 600), color='white')
        visualizations, results = pipeline.predict_image("test_image", test_image)
        
        print("Results (Normalized):")
        print(json.dumps(results, indent=2)) 
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pipeline.cleanup()