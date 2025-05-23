# from ultralytics import YOLO
import os
import io
from pathlib import Path
import base64
from PIL import Image
import cv2
import numpy as np

# %matplotlib inline
# from matplotlib import pyplot as plt

import easyocr
from paddleocr import PaddleOCR

import base64

import torch
from typing import Tuple, List
from torchvision.ops import box_convert
import re
from torchvision.transforms import ToPILImage
import supervision as sv
import torchvision.transforms as T

reader = easyocr.Reader(["en"])
model_dir = Path(__file__).resolve().parent.parent / "models"
paddle_ocr = PaddleOCR(
    lang="en",  # other lang also available
    use_angle_cls=False,
    use_gpu=False,  # using cuda will conflict with pytorch in the same process
    show_log=False,
    max_batch_size=1024,
    use_dilation=True,  # improves accuracy
    det_db_score_mode="slow",  # improves accuracy
    rec_batch_num=1024,
    det_model_dir=str(model_dir / "PaddleOCR" / "det"),
    rec_model_dir=str(model_dir / "PaddleOCR" / "rec"),
    cls_model_dir=str(model_dir / "PaddleOCR" / "cls"),
)


def get_accelerator(no_mps: bool = False) -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and not no_mps:
        return "mps"
    else:
        return "cpu"


def get_yolo_model(model_path):
    from ultralytics import YOLO

    # Load the model.
    model = YOLO(model_path)
    return model


@torch.inference_mode()
def get_parsed_content_icon(
    filtered_boxes, ocr_bbox, image_source, caption_model_processor, prompt=None
):
    to_pil = ToPILImage()
    if ocr_bbox:
        non_ocr_boxes = filtered_boxes[len(ocr_bbox) :]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        xmin, xmax = (
            int(coord[0] * image_source.shape[1]),
            int(coord[2] * image_source.shape[1]),
        )
        ymin, ymax = (
            int(coord[1] * image_source.shape[0]),
            int(coord[3] * image_source.shape[0]),
        )
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        croped_pil_image.append(to_pil(cropped_image))

    model, processor = (
        caption_model_processor["model"],
        caption_model_processor["processor"],
    )
    if not prompt:
        if "florence" in model.config.name_or_path:
            prompt = "<CAPTION>"
        else:
            prompt = "The image shows"

    batch_size = 10  # Number of samples per batch
    generated_texts = []
    device = model.device

    for i in range(0, len(croped_pil_image), batch_size):
        batch = croped_pil_image[i : i + batch_size]
        if model.device.type == "cuda":
            inputs = processor(
                images=batch, text=[prompt] * len(batch), return_tensors="pt"
            ).to(device=device, dtype=torch.float16)
        else:
            inputs = processor(
                images=batch, text=[prompt] * len(batch), return_tensors="pt"
            ).to(device=device)
        if "florence" in model.config.name_or_path:
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )
        else:
            generated_ids = model.generate(
                **inputs,
                max_length=100,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                num_return_sequences=1,
            )  # temperature=0.01, do_sample=True,
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [gen.strip() for gen in generated_text]
        generated_texts.extend(generated_text)

    return generated_texts


def get_parsed_content_icon_phi3v(
    filtered_boxes, ocr_bbox, image_source, caption_model_processor
):
    to_pil = ToPILImage()
    if ocr_bbox:
        non_ocr_boxes = filtered_boxes[len(ocr_bbox) :]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        xmin, xmax = (
            int(coord[0] * image_source.shape[1]),
            int(coord[2] * image_source.shape[1]),
        )
        ymin, ymax = (
            int(coord[1] * image_source.shape[0]),
            int(coord[3] * image_source.shape[0]),
        )
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        croped_pil_image.append(to_pil(cropped_image))

    model, processor = (
        caption_model_processor["model"],
        caption_model_processor["processor"],
    )
    device = model.device
    messages = [
        {"role": "user", "content": "<|image_1|>\ndescribe the icon in one sentence"}
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    batch_size = 5  # Number of samples per batch
    generated_texts = []

    for i in range(0, len(croped_pil_image), batch_size):
        images = croped_pil_image[i : i + batch_size]
        image_inputs = [
            processor.image_processor(x, return_tensors="pt") for x in images
        ]
        inputs = {
            "input_ids": [],
            "attention_mask": [],
            "pixel_values": [],
            "image_sizes": [],
        }
        texts = [prompt] * len(images)
        for i, txt in enumerate(texts):
            input = processor._convert_images_texts_to_inputs(
                image_inputs[i], txt, return_tensors="pt"
            )
            inputs["input_ids"].append(input["input_ids"])
            inputs["attention_mask"].append(input["attention_mask"])
            inputs["pixel_values"].append(input["pixel_values"])
            inputs["image_sizes"].append(input["image_sizes"])
        max_len = max([x.shape[1] for x in inputs["input_ids"]])
        for i, v in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = torch.cat(
                [
                    processor.tokenizer.pad_token_id
                    * torch.ones(1, max_len - v.shape[1], dtype=torch.long),
                    v,
                ],
                dim=1,
            )
            inputs["attention_mask"][i] = torch.cat(
                [
                    torch.zeros(1, max_len - v.shape[1], dtype=torch.long),
                    inputs["attention_mask"][i],
                ],
                dim=1,
            )
        inputs_cat = {k: torch.concatenate(v).to(device) for k, v in inputs.items()}

        generation_args = {
            "max_new_tokens": 25,
            "temperature": 0.01,
            "do_sample": False,
        }
        generate_ids = model.generate(
            **inputs_cat,
            eos_token_id=processor.tokenizer.eos_token_id,
            **generation_args,
        )
        # # remove input tokens
        generate_ids = generate_ids[:, inputs_cat["input_ids"].shape[1] :]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = [res.strip("\n").strip() for res in response]
        generated_texts.extend(response)

    return generated_texts


def remove_overlap(boxes, iou_threshold, ocr_bbox=None):
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1 in enumerate(boxes):
        # if not any(IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2) for j, box2 in enumerate(boxes) if i != j):
        is_valid_box = True
        for j, box2 in enumerate(boxes):
            if (
                i != j
                and IoU(box1, box2) > iou_threshold
                and box_area(box1) > box_area(box2)
            ):
                is_valid_box = False
                break
        if is_valid_box:
            # add the following 2 lines to include ocr bbox
            if ocr_bbox:
                if not any(
                    IoU(box1, box3) > iou_threshold for k, box3 in enumerate(ocr_bbox)
                ):
                    filtered_boxes.append(box1)
            else:
                filtered_boxes.append(box1)
    return torch.tensor(filtered_boxes)


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def annotate(
    image_source: np.ndarray,
    boxes: torch.Tensor,
    phrases: List[str],
    text_scale: float,
    text_padding=5,
    text_thickness=2,
    thickness=3,
) -> np.ndarray:
    """
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates. in cxcywh format, pixel scale
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.
    text_scale (float): The scale of the text to be displayed. 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [f"{phrase}" for phrase in range(boxes.shape[0])]

    from models.OP.util.box_annotator import BoxAnnotator

    box_annotator = BoxAnnotator(
        text_scale=text_scale,
        text_padding=text_padding,
        text_thickness=text_thickness,
        thickness=thickness,
    )  # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels, image_size=(w, h)
    )

    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
    return annotated_frame, label_coordinates


def predict(model, image, caption, box_threshold, text_threshold):
    """Use huggingface model to replace the original model"""
    model, processor = model["model"], model["processor"]
    device = model.device

    inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,  # 0.4,
        text_threshold=text_threshold,  # 0.3,
        target_sizes=[image.size[::-1]],
    )[0]
    boxes, logits, phrases = results["boxes"], results["scores"], results["labels"]
    return boxes, logits, phrases


def predict_yolo(model, image, box_threshold, imgsz):
    """Use huggingface model to replace the original model"""
    # model = model['model']

    result = model.predict(
        source=image,
        conf=box_threshold,
        imgsz=imgsz,
        # iou=0.5, # default 0.7
    )
    boxes = result[0].boxes.xyxy  # .tolist() # in pixel space
    conf = result[0].boxes.conf
    phrases = [str(i) for i in range(len(boxes))]

    return boxes, conf, phrases


def get_som_labeled_img(
    image_source,
    model=None,
    BOX_TRESHOLD=0.01,
    output_coord_in_ratio=False,
    ocr_bbox=None,
    text_scale=0.4,
    text_padding=5,
    draw_bbox_config=None,
    caption_model_processor=None,
    ocr_text=[],
    use_local_semantics=True,
    iou_threshold=0.9,
    prompt=None,
    imgsz=640,
):
    """ocr_bbox: list of xyxy format bbox"""
    w, h = image_source.size
    if model is not None:
        xyxy, logits, phrases = predict_yolo(
            model=model, image=image_source, box_threshold=BOX_TRESHOLD, imgsz=imgsz
        )
        xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
    else:
        xyxy = None

    image_source = np.asarray(image_source)
    

    # annotate the image with labels
    h, w, _ = image_source.shape
    if ocr_bbox:
        ocr_bbox = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
        ocr_bbox = ocr_bbox.tolist()
    else:
        print("no ocr bbox!!!")
        ocr_bbox = None
    
    if xyxy is not None:
        filtered_boxes = remove_overlap(
            boxes=xyxy, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox
        )
    else:
        filtered_boxes = torch.tensor(ocr_bbox)

    # get parsed icon local semantics
    if use_local_semantics:
        caption_model = caption_model_processor["model"]
        if "phi3_v" in caption_model.config.model_type:
            parsed_content_icon = get_parsed_content_icon_phi3v(
                filtered_boxes, ocr_bbox, image_source, caption_model_processor
            )
        else:
            parsed_content_icon = get_parsed_content_icon(
                filtered_boxes,
                ocr_bbox,
                image_source,
                caption_model_processor,
                prompt=prompt,
            )
        ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        icon_start = len(ocr_text)
        parsed_content_icon_ls = []
        for i, txt in enumerate(parsed_content_icon):
            parsed_content_icon_ls.append(f"Icon Box ID {str(i + icon_start)}: {txt}")
        parsed_content_merged = ocr_text + parsed_content_icon_ls
    else:
        ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        parsed_content_merged = ocr_text

    filtered_boxes = box_convert(boxes=filtered_boxes, in_fmt="xyxy", out_fmt="cxcywh")

    phrases = [i for i in range(len(filtered_boxes))]

    # draw boxes
    if draw_bbox_config:
        annotated_frame, label_coordinates = annotate(
            image_source=image_source,
            boxes=filtered_boxes,
            phrases=phrases,
            **draw_bbox_config,
        )
    else:
        annotated_frame, label_coordinates = annotate(
            image_source=image_source,
            boxes=filtered_boxes,
            phrases=phrases,
            text_scale=text_scale,
            text_padding=text_padding,
        )

    pil_img = Image.fromarray(annotated_frame)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("ascii")
    if output_coord_in_ratio:
        # h, w, _ = image_source.shape
        label_coordinates = {
            k: [v[0] / w, v[1] / h, v[2] / w, v[3] / h]
            for k, v in label_coordinates.items()
        }
        assert w == annotated_frame.shape[1] and h == annotated_frame.shape[0]

    return encoded_image, label_coordinates, parsed_content_merged


def get_xywh(input):
    x, y, w, h = (
        input[0][0],
        input[0][1],
        input[2][0] - input[0][0],
        input[2][1] - input[0][1],
    )
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h


def get_xyxy(input):
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp


def get_xywh_yolo(input):
    x, y, w, h = input[0], input[1], input[2] - input[0], input[3] - input[1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h


def check_ocr_box(
    image_path,
    display_img=True,
    output_bb_format="xywh",
    goal_filtering=None,
    easyocr_args=None,
    use_paddleocr=False,
):
    if use_paddleocr:
        result = paddle_ocr.ocr(image_path, cls=False)[0]
        coord = [item[0] for item in result]
        text = [item[1][0] for item in result]
    else:  # EasyOCR
        if easyocr_args is None:
            easyocr_args = {}
        result = reader.readtext(image_path, **easyocr_args)
        # print('goal filtering pred:', result[-5:])
        coord = [item[0] for item in result]
        text = [item[1] for item in result]
    # read the image using cv2
    if display_img:
        opencv_img = cv2.imread(image_path)
        opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
        bb = []
        for item in coord:
            x, y, a, b = get_xywh(item)
            # print(x, y, a, b)
            bb.append((x, y, a, b))
            cv2.rectangle(opencv_img, (x, y), (x + a, y + b), (0, 255, 0), 2)

        # Display the image
        # plt.imshow(opencv_img)
    else:
        if output_bb_format == "xywh":
            bb = [get_xywh(item) for item in coord]
        elif output_bb_format == "xyxy":
            bb = [get_xyxy(item) for item in coord]
        # print('bounding box!!!', bb)
    return (text, bb), goal_filtering


def check_ocr_box_image(
    image, output_bb_format="xywh", goal_filtering=None, easyocr_args=None
):
    if easyocr_args is None:
        easyocr_args = {}

    if type(image) == Image.Image:
        image = np.asarray(image)
    result = reader.readtext(image, **easyocr_args)
    coord = [item[0] for item in result]
    text = [item[1] for item in result]
    if output_bb_format == "xywh":
        bb = [get_xywh(item) for item in coord]
    elif output_bb_format == "xyxy":
        bb = [get_xyxy(item) for item in coord]
    return (text, bb), goal_filtering


def get_image_paths(input_data):
    """
    Loads images from a single image path or a directory containing multiple images.

    Args:
        input_data (str): Path to a single image file or a directory containing image files.

    Returns:
        list: List of paths to all images to be predicted.
    """
    images = []

    if os.path.isdir(input_data):
        # If input_data is a directory, check for nested directories
        for root, dirs, files in os.walk(input_data):
            if dirs:
                raise ValueError(
                    "Input directory should not contain nested directories: {}".format(
                        input_data
                    )
                )
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_path = os.path.join(root, file)
                    images.append(image_path)
            images = sorted(images)
            break  # Only process the top-level directory
    else:
        # Determine the type of input data and process accordingly
        if input_data.lower().endswith((".png", ".jpg", ".jpeg")):
            # If input is a single image file
            images = [input_data]
        else:
            raise ValueError("Unsupported input data format: {}".format(input_data))

    return images


def colormap(N=256, normalized=False):
    """
    Generate the color map.

    Args:
        N (int): Number of labels (default is 256).
        normalized (bool): If True, return colors normalized to [0, 1]. Otherwise, return [0, 255].

    Returns:
        np.ndarray: Color map array of shape (N, 3).
    """

    def bitget(byteval, idx):
        """
        Get the bit value at the specified index.

        Args:
            byteval (int): The byte value.
            idx (int): The index of the bit.

        Returns:
            int: The bit value (0 or 1).
        """
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << (7 - j))
            g = g | (bitget(c, 1) << (7 - j))
            b = b | (bitget(c, 2) << (7 - j))
            c = c >> 3
        cmap[i] = np.array([r, g, b])

    if normalized:
        cmap = cmap.astype(np.float32) / 255.0

    return cmap


def visualize_bbox(image_path, bboxes, classes, scores, id_to_names, alpha=0.3):
    """
    Visualize layout detection results on an image.

    Args:
        image_path (str): Path to the input image.
        bboxes (list): List of bounding boxes, each represented as [x_min, y_min, x_max, y_max].
        classes (list): List of class IDs corresponding to the bounding boxes.
        id_to_names (dict): Dictionary mapping class IDs to class names.
        alpha (float): Transparency factor for the filled color (default is 0.3).

    Returns:
        np.ndarray: Image with visualized layout detection results.
    """
    # Check if image_path is a PIL.Image.Image object
    if isinstance(image_path, Image.Image):
        image = np.array(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    else:
        image = cv2.imread(image_path)

    overlay = image.copy()

    cmap = colormap(N=len(id_to_names), normalized=False)

    # Iterate over each bounding box
    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = map(int, bbox)
        class_id = int(classes[i])
        class_name = id_to_names[class_id]

        text = class_name + f":{scores[i]:.3f}"

        color = tuple(int(c) for c in cmap[class_id])
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # Add the class name with a background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
        )
        cv2.rectangle(
            image,
            (x_min, y_min - text_height - baseline),
            (x_min + text_width, y_min),
            color,
            -1,
        )
        cv2.putText(
            image,
            text,
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    return image
