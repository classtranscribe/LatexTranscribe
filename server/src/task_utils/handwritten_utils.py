import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

import pyclipper
from shapely.geometry import Polygon

from src.task_utils.classification import textnet_base, textnet_small, textnet_tiny
from src.task_utils.utils import _bf16_to_float32, load_pretrained_params
from src.task_utils.modules.layers import FASTConvLayer
from typing import *

CLASS_NAME: str = "words"

default_cfgs: dict[str, dict[str, Any]] = {
    "fast_tiny": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://doctr-static.mindee.com/models?id=v0.8.1/fast_tiny-1acac421.pt&src=0",
    },
    "fast_small": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://doctr-static.mindee.com/models?id=v0.8.1/fast_small-10952cc1.pt&src=0",
    },
    "fast_base": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://doctr-static.mindee.com/models?id=v0.8.1/fast_base-688a8b34.pt&src=0",
    },
}

def _addindent(s_, num_spaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class NestedObject:
    """Base class for all nested objects in doctr"""

    _children_names: list[str]

    def extra_repr(self) -> str:
        return ""

    def __repr__(self):
        # We treat the extra repr like the sub-object, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        if hasattr(self, "_children_names"):
            for key in self._children_names:
                child = getattr(self, key)
                if isinstance(child, list) and len(child) > 0:
                    child_str = ",\n".join([repr(subchild) for subchild in child])
                    if len(child) > 1:
                        child_str = _addindent(f"\n{child_str},", 2) + "\n"
                    child_str = f"[{child_str}]"
                else:
                    child_str = repr(child)
                child_str = _addindent(child_str, 2)
                child_lines.append("(" + key + "): " + child_str)
        lines = extra_lines + child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

class BaseModel(NestedObject):
    """Implements abstract DetectionModel class"""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.cfg = cfg

class DetectionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
        box_thresh (float): minimal objectness score to consider a box
        bin_thresh (float): threshold to apply to segmentation raw heatmap
        assume straight_pages (bool): if True, fit straight boxes only
    """

    def __init__(self, box_thresh: float = 0.5, bin_thresh: float = 0.5, assume_straight_pages: bool = True) -> None:
        self.box_thresh = box_thresh
        self.bin_thresh = bin_thresh
        self.assume_straight_pages = assume_straight_pages
        self._opening_kernel: np.ndarray = np.ones((3, 3), dtype=np.uint8)

    def extra_repr(self) -> str:
        return f"bin_thresh={self.bin_thresh}, box_thresh={self.box_thresh}"

    @staticmethod
    def box_score(pred: np.ndarray, points: np.ndarray, assume_straight_pages: bool = True) -> float:
        """Compute the confidence score for a polygon : mean of the p values on the polygon

        Args:
            pred (np.ndarray): p map returned by the model
            points: coordinates of the polygon
            assume_straight_pages: if True, fit straight boxes only

        Returns:
            polygon objectness
        """
        h, w = pred.shape[:2]

        if assume_straight_pages:
            xmin = np.clip(np.floor(points[:, 0].min()).astype(np.int32), 0, w - 1)
            xmax = np.clip(np.ceil(points[:, 0].max()).astype(np.int32), 0, w - 1)
            ymin = np.clip(np.floor(points[:, 1].min()).astype(np.int32), 0, h - 1)
            ymax = np.clip(np.ceil(points[:, 1].max()).astype(np.int32), 0, h - 1)
            return pred[ymin : ymax + 1, xmin : xmax + 1].mean()

        else:
            mask: np.ndarray = np.zeros((h, w), np.int32)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1.0)  # type: ignore[call-overload]
            product = pred * mask
            return np.sum(product) / np.count_nonzero(product)

    def bitmap_to_boxes(
        self,
        pred: np.ndarray,
        bitmap: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def __call__(
        self,
        proba_map,
    ) -> list[list[np.ndarray]]:
        """Performs postprocessing for a list of model outputs

        Args:
            proba_map: probability map of shape (N, H, W, C)

        Returns:
            list of N class predictions (for each input sample), where each class predictions is a list of C tensors
        of shape (*, 5) or (*, 6)
        """
        if proba_map.ndim != 4:
            raise AssertionError(f"arg `proba_map` is expected to be 4-dimensional, got {proba_map.ndim}.")

        # Erosion + dilation on the binary map
        bin_map = [
            [
                cv2.morphologyEx(bmap[..., idx], cv2.MORPH_OPEN, self._opening_kernel)
                for idx in range(proba_map.shape[-1])
            ]
            for bmap in (proba_map >= self.bin_thresh).astype(np.uint8)
        ]

        return [
            [self.bitmap_to_boxes(pmaps[..., idx], bmaps[idx]) for idx in range(proba_map.shape[-1])]
            for pmaps, bmaps in zip(proba_map, bin_map)
        ]

class FASTPostProcessor(DetectionPostProcessor):
    """Implements a post processor for FAST model.

    Args:
        bin_thresh: threshold used to binzarized p_map at inference time
        box_thresh: minimal objectness score to consider a box
        assume_straight_pages: whether the inputs were expected to have horizontal text elements
    """

    def __init__(
        self,
        bin_thresh: float = 0.1,
        box_thresh: float = 0.1,
        assume_straight_pages: bool = True,
    ) -> None:
        super().__init__(box_thresh, bin_thresh, assume_straight_pages)
        self.unclip_ratio = 1.0

    def polygon_to_box(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """Expand a polygon (points) by a factor unclip_ratio, and returns a polygon

        Args:
            points: The first parameter.

        Returns:
            a box in absolute coordinates (xmin, ymin, xmax, ymax) or (4, 2) array (quadrangle)
        """
        if not self.assume_straight_pages:
            # Compute the rectangle polygon enclosing the raw polygon
            rect = cv2.minAreaRect(points)
            points = cv2.boxPoints(rect)
            # Add 1 pixel to correct cv2 approx
            area = (rect[1][0] + 1) * (1 + rect[1][1])
            length = 2 * (rect[1][0] + rect[1][1]) + 2
        else:
            poly = Polygon(points)
            area = poly.area
            length = poly.length
        distance = area * self.unclip_ratio / length  # compute distance to expand polygon
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        _points = offset.Execute(distance)
        # Take biggest stack of points
        idx = 0
        if len(_points) > 1:
            max_size = 0
            for _idx, p in enumerate(_points):
                if len(p) > max_size:
                    idx = _idx
                    max_size = len(p)
            # We ensure that _points can be correctly casted to a ndarray
            _points = [_points[idx]]
        expanded_points: np.ndarray = np.asarray(_points)  # expand polygon
        if len(expanded_points) < 1:
            return None  # type: ignore[return-value]
        return (
            cv2.boundingRect(expanded_points)  # type: ignore[return-value]
            if self.assume_straight_pages
            else np.roll(cv2.boxPoints(cv2.minAreaRect(expanded_points)), -1, axis=0)
        )

    def bitmap_to_boxes(
        self,
        pred: np.ndarray,
        bitmap: np.ndarray,
    ) -> np.ndarray:
        """Compute boxes from a bitmap/pred_map: find connected components then filter boxes

        Args:
            pred: Pred map from differentiable linknet output
            bitmap: Bitmap map computed from pred (binarized)
            angle_tol: Comparison tolerance of the angle with the median angle across the page
            ratio_tol: Under this limit aspect ratio, we cannot resolve the direction of the crop

        Returns:
            np tensor boxes for the bitmap, each box is a 6-element list
                containing x, y, w, h, alpha, score for the box
        """
        height, width = bitmap.shape[:2]
        boxes: list[np.ndarray | list[float]] = []
        # get contours from connected components on the bitmap
        contours, _ = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Check whether smallest enclosing bounding box is not too small
            if np.any(contour[:, 0].max(axis=0) - contour[:, 0].min(axis=0) < 2):
                continue
            # Compute objectness
            if self.assume_straight_pages:
                x, y, w, h = cv2.boundingRect(contour)
                points: np.ndarray = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])
                score = self.box_score(pred, points, assume_straight_pages=True)
            else:
                score = self.box_score(pred, contour, assume_straight_pages=False)

            if score < self.box_thresh:  # remove polygons with a weak objectness
                continue

            if self.assume_straight_pages:
                _box = self.polygon_to_box(points)
            else:
                _box = self.polygon_to_box(np.squeeze(contour))

            if self.assume_straight_pages:
                # compute relative polygon to get rid of img shape
                x, y, w, h = _box
                xmin, ymin, xmax, ymax = x / width, y / height, (x + w) / width, (y + h) / height
                boxes.append([xmin, ymin, xmax, ymax, score])
            else:
                # compute relative box to get rid of img shape
                _box[:, 0] /= width
                _box[:, 1] /= height
                # Add score to box as (0, score)
                boxes.append(np.vstack([_box, np.array([0.0, score])]))

        if not self.assume_straight_pages:
            return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 5, 2), dtype=pred.dtype)
        else:
            return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 5), dtype=pred.dtype)

class _FAST(BaseModel):
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_.
    """

    min_size_box: int = 3
    assume_straight_pages: bool = True
    shrink_ratio = 0.4

    def build_target(
        self,
        target: list[dict[str, np.ndarray]],
        output_shape: tuple[int, int, int],
        channels_last: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the target, and it's mask to be used from loss computation.

        Args:
            target: target coming from dataset
            output_shape: shape of the output of the model without batch_size
            channels_last: whether channels are last or not

        Returns:
            the new formatted target, mask and shrunken text kernel
        """
        if any(t.dtype != np.float32 for tgt in target for t in tgt.values()):
            raise AssertionError("the expected dtype of target 'boxes' entry is 'np.float32'.")
        if any(np.any((t[:, :4] > 1) | (t[:, :4] < 0)) for tgt in target for t in tgt.values()):
            raise ValueError("the 'boxes' entry of the target is expected to take values between 0 & 1.")

        h: int
        w: int
        if channels_last:
            h, w, num_classes = output_shape
        else:
            num_classes, h, w = output_shape
        target_shape = (len(target), num_classes, h, w)

        seg_target: np.ndarray = np.zeros(target_shape, dtype=np.uint8)
        seg_mask: np.ndarray = np.ones(target_shape, dtype=bool)
        shrunken_kernel: np.ndarray = np.zeros(target_shape, dtype=np.uint8)

        for idx, tgt in enumerate(target):
            for class_idx, _tgt in enumerate(tgt.values()):
                # Draw each polygon on gt
                if _tgt.shape[0] == 0:
                    # Empty image, full masked
                    seg_mask[idx, class_idx] = False

                # Absolute bounding boxes
                abs_boxes = _tgt.copy()

                if abs_boxes.ndim == 3:
                    abs_boxes[:, :, 0] *= w
                    abs_boxes[:, :, 1] *= h
                    polys = abs_boxes
                    boxes_size = np.linalg.norm(abs_boxes[:, 2, :] - abs_boxes[:, 0, :], axis=-1)
                    abs_boxes = np.concatenate((abs_boxes.min(1), abs_boxes.max(1)), -1).round().astype(np.int32)
                else:
                    abs_boxes[:, [0, 2]] *= w
                    abs_boxes[:, [1, 3]] *= h
                    abs_boxes = abs_boxes.round().astype(np.int32)
                    polys = np.stack(
                        [
                            abs_boxes[:, [0, 1]],
                            abs_boxes[:, [0, 3]],
                            abs_boxes[:, [2, 3]],
                            abs_boxes[:, [2, 1]],
                        ],
                        axis=1,
                    )
                    boxes_size = np.minimum(abs_boxes[:, 2] - abs_boxes[:, 0], abs_boxes[:, 3] - abs_boxes[:, 1])

                for poly, box, box_size in zip(polys, abs_boxes, boxes_size):
                    # Mask boxes that are too small
                    if box_size < self.min_size_box:
                        seg_mask[idx, class_idx, box[1] : box[3] + 1, box[0] : box[2] + 1] = False
                        continue

                    # Negative shrink for gt, as described in paper
                    polygon = Polygon(poly)
                    distance = polygon.area * (1 - np.power(self.shrink_ratio, 2)) / polygon.length
                    subject = [tuple(coor) for coor in poly]
                    padding = pyclipper.PyclipperOffset()
                    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                    shrunken = padding.Execute(-distance)

                    # Draw polygon on gt if it is valid
                    if len(shrunken) == 0:
                        seg_mask[idx, class_idx, box[1] : box[3] + 1, box[0] : box[2] + 1] = False
                        continue
                    shrunken = np.array(shrunken[0]).reshape(-1, 2)
                    if shrunken.shape[0] <= 2 or not Polygon(shrunken).is_valid:
                        seg_mask[idx, class_idx, box[1] : box[3] + 1, box[0] : box[2] + 1] = False
                        continue
                    cv2.fillPoly(shrunken_kernel[idx, class_idx], [shrunken.astype(np.int32)], 1.0)  # type: ignore[call-overload]
                    # draw the original polygon on the segmentation target
                    cv2.fillPoly(seg_target[idx, class_idx], [poly.astype(np.int32)], 1.0)  # type: ignore[call-overload]

        # Don't forget to switch back to channel last if Tensorflow is used
        if channels_last:
            seg_target = seg_target.transpose((0, 2, 3, 1))
            seg_mask = seg_mask.transpose((0, 2, 3, 1))
            shrunken_kernel = shrunken_kernel.transpose((0, 2, 3, 1))

        return seg_target, seg_mask, shrunken_kernel

class FastNeck(nn.Module):
    """Neck of the FAST architecture, composed of a series of 3x3 convolutions and upsampling layers.

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 128,
    ) -> None:
        super().__init__()
        self.reduction = nn.ModuleList([
            FASTConvLayer(in_channels * scale, out_channels, kernel_size=3) for scale in [1, 2, 4, 8]
        ])

    def _upsample(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=y.shape[-2:], mode="bilinear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1, f2, f3, f4 = x
        f1, f2, f3, f4 = [reduction(f) for reduction, f in zip(self.reduction, (f1, f2, f3, f4))]
        f2, f3, f4 = [self._upsample(f, f1) for f in (f2, f3, f4)]
        f = torch.cat((f1, f2, f3, f4), 1)
        return f

class FastHead(nn.Sequential):
    """Head of the FAST architecture

    Args:
        in_channels: number of input channels
        num_classes: number of output classes
        out_channels: number of output channels
        dropout: dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        out_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        _layers: list[nn.Module] = [
            FASTConvLayer(in_channels, out_channels, kernel_size=3),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, num_classes, kernel_size=1, bias=False),
        ]
        super().__init__(*_layers)

class FAST(_FAST, nn.Module):
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_.

    Args:
        feat extractor: the backbone serving as feature extractor
        bin_thresh: threshold for binarization
        box_thresh: minimal objectness score to consider a box
        dropout_prob: dropout probability
        pooling_size: size of the pooling layer
        assume_straight_pages: if True, fit straight bounding boxes only
        exportable: onnx exportable returns only logits
        cfg: the configuration dict of the model
        class_names: list of class names
    """

    def __init__(
        self,
        feat_extractor: IntermediateLayerGetter,
        bin_thresh: float = 0.1,
        box_thresh: float = 0.1,
        dropout_prob: float = 0.1,
        pooling_size: int = 4,  # different from paper performs better on close text-rich images
        assume_straight_pages: bool = True,
        exportable: bool = False,
        cfg: dict[str, Any] = {},
        class_names: list[str] = [CLASS_NAME],
    ) -> None:
        super().__init__()
        self.class_names = class_names
        num_classes: int = len(self.class_names)
        self.cfg = cfg

        self.exportable = exportable
        self.assume_straight_pages = assume_straight_pages

        self.feat_extractor = feat_extractor
        # Identify the number of channels for the neck & head initialization
        _is_training = self.feat_extractor.training
        self.feat_extractor = self.feat_extractor.eval()
        with torch.no_grad():
            out = self.feat_extractor(torch.zeros((1, 3, 32, 32)))
            feat_out_channels = [v.shape[1] for _, v in out.items()]

        if _is_training:
            self.feat_extractor = self.feat_extractor.train()

        # Initialize neck & head
        self.neck = FastNeck(feat_out_channels[0], feat_out_channels[1])
        self.prob_head = FastHead(feat_out_channels[-1], num_classes, feat_out_channels[1], dropout_prob)

        # NOTE: The post processing from the paper works not well for text-rich images
        # so we use a modified version from DBNet
        self.postprocessor = FASTPostProcessor(
            assume_straight_pages=assume_straight_pages, bin_thresh=bin_thresh, box_thresh=box_thresh
        )

        # Pooling layer as erosion reversal as described in the paper
        self.pooling = nn.MaxPool2d(kernel_size=pooling_size // 2 + 1, stride=1, padding=(pooling_size // 2) // 2)

        for n, m in self.named_modules():
            # Don't override the initialization of the backbone
            if n.startswith("feat_extractor."):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        target: list[np.ndarray] | None = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> dict[str, torch.Tensor]:
        # Extract feature maps at different stages
        feats = self.feat_extractor(x)
        feats = [feats[str(idx)] for idx in range(len(feats))]
        # Pass through the Neck & Head & Upsample
        feat_concat = self.neck(feats)
        logits = F.interpolate(self.prob_head(feat_concat), size=x.shape[-2:], mode="bilinear")

        out: dict[str, Any] = {}
        if self.exportable:
            out["logits"] = logits
            return out

        if return_model_output or target is None or return_preds:
            prob_map = _bf16_to_float32(torch.sigmoid(self.pooling(logits)))

        if return_model_output:
            out["out_map"] = prob_map

        if target is None or return_preds:
            # Disable for torch.compile compatibility
            @torch.compiler.disable  # type: ignore[attr-defined]
            def _postprocess(prob_map: torch.Tensor) -> list[dict[str, Any]]:
                return [
                    dict(zip(self.class_names, preds))
                    for preds in self.postprocessor(prob_map.detach().cpu().permute((0, 2, 3, 1)).numpy())
                ]

            # Post-process boxes (keep only text predictions)
            out["preds"] = _postprocess(prob_map)

        if target is not None:
            loss = self.compute_loss(logits, target)
            out["loss"] = loss

        return out

    def compute_loss(
        self,
        out_map: torch.Tensor,
        target: list[np.ndarray],
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute fast loss, 2 x Dice loss where the text kernel loss is scaled by 0.5.

        Args:
            out_map: output feature map of the model of shape (N, num_classes, H, W)
            target: list of dictionary where each dict has a `boxes` and a `flags` entry
            eps: epsilon factor in dice loss

        Returns:
            A loss tensor
        """
        targets = self.build_target(target, out_map.shape[1:], False)  # type: ignore[arg-type]

        seg_target, seg_mask = torch.from_numpy(targets[0]), torch.from_numpy(targets[1])
        shrunken_kernel = torch.from_numpy(targets[2]).to(out_map.device)
        seg_target, seg_mask = seg_target.to(out_map.device), seg_mask.to(out_map.device)

        def ohem_sample(score: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            masks = []
            for class_idx in range(gt.shape[0]):
                pos_num = int(torch.sum(gt[class_idx] > 0.5)) - int(
                    torch.sum((gt[class_idx] > 0.5) & (mask[class_idx] <= 0.5))
                )
                neg_num = int(torch.sum(gt[class_idx] <= 0.5))
                neg_num = int(min(pos_num * 3, neg_num))

                if neg_num == 0 or pos_num == 0:
                    masks.append(mask[class_idx])
                    continue

                neg_score_sorted, _ = torch.sort(-score[class_idx][gt[class_idx] <= 0.5])
                threshold = -neg_score_sorted[neg_num - 1]

                selected_mask = ((score[class_idx] >= threshold) | (gt[class_idx] > 0.5)) & (mask[class_idx] > 0.5)
                masks.append(selected_mask)
            # combine all masks to shape (len(masks), H, W)
            return torch.stack(masks).unsqueeze(0).float()

        if len(self.class_names) > 1:
            kernels = torch.softmax(out_map, dim=1)
            prob_map = torch.softmax(self.pooling(out_map), dim=1)
        else:
            kernels = torch.sigmoid(out_map)
            prob_map = torch.sigmoid(self.pooling(out_map))

        # As described in the paper, we use the Dice loss for the text segmentation map and the Dice loss scaled by 0.5.
        selected_masks = torch.cat(
            [ohem_sample(score, gt, mask) for score, gt, mask in zip(prob_map, seg_target, seg_mask)], 0
        ).float()
        inter = (selected_masks * prob_map * seg_target).sum((0, 2, 3))
        cardinality = (selected_masks * (prob_map + seg_target)).sum((0, 2, 3))
        text_loss = (1 - 2 * inter / (cardinality + eps)).mean() * 0.5

        # As described in the paper, we use the Dice loss for the text kernel map.
        selected_masks = seg_target * seg_mask
        inter = (selected_masks * kernels * shrunken_kernel).sum((0, 2, 3))  # noqa
        cardinality = (selected_masks * (kernels + shrunken_kernel)).sum((0, 2, 3))  # noqa
        kernel_loss = (1 - 2 * inter / (cardinality + eps)).mean()

        return text_loss + kernel_loss

def reparameterize(model: FAST | nn.Module) -> FAST:
    """Fuse batchnorm and conv layers and reparameterize the model

    Args:
        model: the FAST model to reparameterize

    Returns:
        the reparameterized model
    """
    last_conv = None
    last_conv_name = None

    for module in model.modules():
        if hasattr(module, "reparameterize_layer"):
            module.reparameterize_layer()

    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # fuse batchnorm only if it is followed by a conv layer
            if last_conv is None:
                continue
            conv_w = last_conv.weight
            conv_b = last_conv.bias if last_conv.bias is not None else torch.zeros_like(child.running_mean)  # type: ignore[arg-type]

            factor = child.weight / torch.sqrt(child.running_var + child.eps)  # type: ignore
            last_conv.weight = nn.Parameter(conv_w * factor.reshape([last_conv.out_channels, 1, 1, 1]))
            last_conv.bias = nn.Parameter((conv_b - child.running_mean) * factor + child.bias)
            model._modules[last_conv_name] = last_conv  # type: ignore[index]
            model._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            reparameterize(child)

    return model  # type: ignore[return-value]

def _fast(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[bool], nn.Module],
    feat_layers: list[str],
    pretrained_backbone: bool = True,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> FAST:
    pretrained_backbone = pretrained_backbone and not pretrained

    # Build the feature extractor
    feat_extractor = IntermediateLayerGetter(
        backbone_fn(pretrained_backbone),
        {layer_name: str(idx) for idx, layer_name in enumerate(feat_layers)},
    )

    if not kwargs.get("class_names", None):
        kwargs["class_names"] = default_cfgs[arch].get("class_names", [CLASS_NAME])
    else:
        kwargs["class_names"] = sorted(kwargs["class_names"])
    # Build the model
    model = FAST(feat_extractor, cfg=default_cfgs[arch], **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of class_names is not the same as the number of classes in the pretrained model =>
        # remove the layer weights
        _ignore_keys = (
            ignore_keys if kwargs["class_names"] != default_cfgs[arch].get("class_names", [CLASS_NAME]) else None
        )
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model

def fast_tiny(pretrained: bool = False, **kwargs: Any) -> FAST:
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_, using a tiny TextNet backbone.

    >>> import torch
    >>> from doctr.models import fast_tiny
    >>> model = fast_tiny(pretrained=True)
    >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
        text detection architecture
    """
    return _fast(
        "fast_tiny",
        pretrained,
        textnet_tiny,
        ["3", "4", "5", "6"],
        ignore_keys=["prob_head.2.weight"],
        **kwargs,
    )

def fast_small(pretrained: bool = False, **kwargs: Any) -> FAST:
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_, using a small TextNet backbone.

    >>> import torch
    >>> from doctr.models import fast_small
    >>> model = fast_small(pretrained=True)
    >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
        text detection architecture
    """
    return _fast(
        "fast_small",
        pretrained,
        textnet_small,
        ["3", "4", "5", "6"],
        ignore_keys=["prob_head.2.weight"],
        **kwargs,
    )

def fast_base(pretrained: bool = False, **kwargs: Any) -> FAST:
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_, using a base TextNet backbone.

    >>> import torch
    >>> from doctr.models import fast_base
    >>> model = fast_base(pretrained=True)
    >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
        text detection architecture
    """
    return _fast(
        "fast_base",
        pretrained,
        textnet_base,
        ["3", "4", "5", "6"],
        ignore_keys=["prob_head.2.weight"],
        **kwargs,
    )


def _remove_padding(
    pages: list[np.ndarray],
    loc_preds: list[dict[str, np.ndarray]],
    preserve_aspect_ratio: bool,
    symmetric_pad: bool,
    assume_straight_pages: bool,
) -> list[dict[str, np.ndarray]]:
    """Remove padding from the localization predictions

    Args:
        pages: list of pages
        loc_preds: list of localization predictions
        preserve_aspect_ratio: whether the aspect ratio was preserved during padding
        symmetric_pad: whether the padding was symmetric
        assume_straight_pages: whether the pages are assumed to be straight

    Returns:
        list of unpaded localization predictions
    """
    if preserve_aspect_ratio:
        # Rectify loc_preds to remove padding
        rectified_preds = []
        for page, dict_loc_preds in zip(pages, loc_preds):
            for k, loc_pred in dict_loc_preds.items():
                h, w = page.shape[0], page.shape[1]
                if h > w:
                    # y unchanged, dilate x coord
                    if symmetric_pad:
                        if assume_straight_pages:
                            loc_pred[:, [0, 2]] = (loc_pred[:, [0, 2]] - 0.5) * h / w + 0.5
                        else:
                            loc_pred[:, :, 0] = (loc_pred[:, :, 0] - 0.5) * h / w + 0.5
                    else:
                        if assume_straight_pages:
                            loc_pred[:, [0, 2]] *= h / w
                        else:
                            loc_pred[:, :, 0] *= h / w
                elif w > h:
                    # x unchanged, dilate y coord
                    if symmetric_pad:
                        if assume_straight_pages:
                            loc_pred[:, [1, 3]] = (loc_pred[:, [1, 3]] - 0.5) * w / h + 0.5
                        else:
                            loc_pred[:, :, 1] = (loc_pred[:, :, 1] - 0.5) * w / h + 0.5
                    else:
                        if assume_straight_pages:
                            loc_pred[:, [1, 3]] *= w / h
                        else:
                            loc_pred[:, :, 1] *= w / h
                rectified_preds.append({k: np.clip(loc_pred, 0, 1)})
        return rectified_preds
    return loc_preds

def set_device_and_dtype(
    model: Any, batches: list[torch.Tensor], device: str | torch.device, dtype: torch.dtype
) -> tuple[Any, list[torch.Tensor]]:
    """Set the device and dtype of a model and its batches

    >>> import torch
    >>> from torch import nn
    >>> from doctr.models.utils import set_device_and_dtype
    >>> model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    >>> batches = [torch.rand(8) for _ in range(2)]
    >>> model, batches = set_device_and_dtype(model, batches, device="cuda", dtype=torch.float16)

    Args:
        model: the model to be set
        batches: the batches to be set
        device: the device to be used
        dtype: the dtype to be used

    Returns:
        the model and batches set
    """
    return model.to(device=device, dtype=dtype), [batch.to(device=device, dtype=dtype) for batch in batches]

if __name__ == "__main__":
    model = fast_base(pretrained=True)
    input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    out = model(input_tensor)
    print(out)