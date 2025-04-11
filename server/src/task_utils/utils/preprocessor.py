# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.functional import pad
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
import multiprocessing as mp
from collections.abc import Callable, Iterable, Iterator
from multiprocessing.pool import ThreadPool
from typing import Any

__all__ = ["PreProcessor"]


def multithread_exec(func: Callable[[Any], Any], seq: Iterable[Any], threads: int | None = None) -> Iterator[Any]:
    """Execute a given function in parallel for each element of a given sequence

    >>> from doctr.utils.multithreading import multithread_exec
    >>> entries = [1, 4, 8]
    >>> results = multithread_exec(lambda x: x ** 2, entries)

    Args:
        func: function to be executed on each element of the iterable
        seq: iterable
        threads: number of workers to be used for multiprocessing

    Returns:
        iterator of the function's results using the iterable as inputs

    Notes:
        This function uses ThreadPool from multiprocessing package, which uses `/dev/shm` directory for shared memory.
        If you do not have write permissions for this directory (if you run `doctr` on AWS Lambda for instance),
        you might want to disable multiprocessing. To achieve that, set 'DOCTR_MULTIPROCESSING_DISABLE' to 'TRUE'.
    """
    threads = threads if isinstance(threads, int) else min(16, mp.cpu_count())
    # Single-thread
    if threads < 2:
        results = map(func, seq)
    # Multi-threading
    else:
        with ThreadPool(threads) as tp:
            # ThreadPool's map function returns a list, but seq could be of a different type
            # That's why wrapping result in map to return iterator
            results = map(lambda x: x, tp.map(func, seq))  # noqa: C417
    return results

class Resize(T.Resize):
    """Resize the input image to the given size"""

    def __init__(
        self,
        size: int | tuple[int, int],
        interpolation=F.InterpolationMode.BILINEAR,
        preserve_aspect_ratio: bool = False,
        symmetric_pad: bool = False,
    ) -> None:
        super().__init__(size, interpolation, antialias=True)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad

        if not isinstance(self.size, (int, tuple, list)):
            raise AssertionError("size should be either a tuple, a list or an int")

    def forward(
        self,
        img: torch.Tensor,
        target: np.ndarray | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, np.ndarray]:
        if isinstance(self.size, int):
            target_ratio = img.shape[-2] / img.shape[-1]
        else:
            target_ratio = self.size[0] / self.size[1]
        actual_ratio = img.shape[-2] / img.shape[-1]

        if not self.preserve_aspect_ratio or (target_ratio == actual_ratio and (isinstance(self.size, (tuple, list)))):
            # If we don't preserve the aspect ratio or the wanted aspect ratio is the same than the original one
            # We can use with the regular resize
            if target is not None:
                return super().forward(img), target
            return super().forward(img)
        else:
            # Resize
            if isinstance(self.size, (tuple, list)):
                if actual_ratio > target_ratio:
                    tmp_size = (self.size[0], max(int(self.size[0] / actual_ratio), 1))
                else:
                    tmp_size = (max(int(self.size[1] * actual_ratio), 1), self.size[1])
            elif isinstance(self.size, int):  # self.size is the longest side, infer the other
                if img.shape[-2] <= img.shape[-1]:
                    tmp_size = (max(int(self.size * actual_ratio), 1), self.size)
                else:
                    tmp_size = (self.size, max(int(self.size / actual_ratio), 1))

            # Scale image
            img = F.resize(img, tmp_size, self.interpolation, antialias=True)
            raw_shape = img.shape[-2:]
            if isinstance(self.size, (tuple, list)):
                # Pad (inverted in pytorch)
                _pad = (0, self.size[1] - img.shape[-1], 0, self.size[0] - img.shape[-2])
                if self.symmetric_pad:
                    half_pad = (math.ceil(_pad[1] / 2), math.ceil(_pad[3] / 2))
                    _pad = (half_pad[0], _pad[1] - half_pad[0], half_pad[1], _pad[3] - half_pad[1])
                # Pad image
                img = pad(img, _pad)

            # In case boxes are provided, resize boxes if needed (for detection task if preserve aspect ratio)
            if target is not None:
                if self.symmetric_pad:
                    offset = half_pad[0] / img.shape[-1], half_pad[1] / img.shape[-2]

                if self.preserve_aspect_ratio:
                    # Get absolute coords
                    if target.shape[1:] == (4,):
                        if isinstance(self.size, (tuple, list)) and self.symmetric_pad:
                            target[:, [0, 2]] = offset[0] + target[:, [0, 2]] * raw_shape[-1] / img.shape[-1]
                            target[:, [1, 3]] = offset[1] + target[:, [1, 3]] * raw_shape[-2] / img.shape[-2]
                        else:
                            target[:, [0, 2]] *= raw_shape[-1] / img.shape[-1]
                            target[:, [1, 3]] *= raw_shape[-2] / img.shape[-2]
                    elif target.shape[1:] == (4, 2):
                        if isinstance(self.size, (tuple, list)) and self.symmetric_pad:
                            target[..., 0] = offset[0] + target[..., 0] * raw_shape[-1] / img.shape[-1]
                            target[..., 1] = offset[1] + target[..., 1] * raw_shape[-2] / img.shape[-2]
                        else:
                            target[..., 0] *= raw_shape[-1] / img.shape[-1]
                            target[..., 1] *= raw_shape[-2] / img.shape[-2]
                    else:
                        raise AssertionError("Boxes should be in the format (n_boxes, 4, 2) or (n_boxes, 4)")

                return img, np.clip(target, 0, 1)

            return img

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        _repr = f"output_size={self.size}, interpolation='{interpolate_str}'"
        if self.preserve_aspect_ratio:
            _repr += f", preserve_aspect_ratio={self.preserve_aspect_ratio}, symmetric_pad={self.symmetric_pad}"
        return f"{self.__class__.__name__}({_repr})"

class PreProcessor(nn.Module):
    """Implements an abstract preprocessor object which performs casting, resizing, batching and normalization.

    Args:
        output_size: expected size of each page in format (H, W)
        batch_size: the size of page batches
        mean: mean value of the training distribution by channel
        std: standard deviation of the training distribution by channel
        **kwargs: additional arguments for the resizing operation
    """

    def __init__(
        self,
        output_size: tuple[int, int],
        batch_size: int,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.resize: T.Resize = Resize(output_size, **kwargs)
        # Perform the division by 255 at the same time
        self.normalize = T.Normalize(mean, std)

    def batch_inputs(self, samples: list[torch.Tensor]) -> list[torch.Tensor]:
        """Gather samples into batches for inference purposes

        Args:
            samples: list of samples of shape (C, H, W)

        Returns:
            list of batched samples (*, C, H, W)
        """
        num_batches = int(math.ceil(len(samples) / self.batch_size))
        batches = [
            torch.stack(samples[idx * self.batch_size : min((idx + 1) * self.batch_size, len(samples))], dim=0)
            for idx in range(int(num_batches))
        ]

        return batches

    def sample_transforms(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise AssertionError("expected list of 3D Tensors")
        if isinstance(x, np.ndarray):
            if x.dtype not in (np.uint8, np.float32):
                raise TypeError("unsupported data type for numpy.ndarray")
            x = torch.from_numpy(x.copy()).permute(2, 0, 1)
        elif x.dtype not in (torch.uint8, torch.float16, torch.float32):
            raise TypeError("unsupported data type for torch.Tensor")
        # Resizing
        x = self.resize(x)
        # Data type
        if x.dtype == torch.uint8:
            x = x.to(dtype=torch.float32).div(255).clip(0, 1)  # type: ignore[union-attr]
        else:
            x = x.to(dtype=torch.float32)  # type: ignore[union-attr]

        return x  # type: ignore[return-value]

    def __call__(self, x: torch.Tensor | np.ndarray | list[torch.Tensor | np.ndarray]) -> list[torch.Tensor]:
        """Prepare document data for model forwarding

        Args:
            x: list of images (np.array) or tensors (already resized and batched)

        Returns:
            list of page batches
        """
        # Input type check
        if isinstance(x, (np.ndarray, torch.Tensor)):
            if x.ndim != 4:
                raise AssertionError("expected 4D Tensor")
            if isinstance(x, np.ndarray):
                if x.dtype not in (np.uint8, np.float32):
                    raise TypeError("unsupported data type for numpy.ndarray")
                x = torch.from_numpy(x.copy()).permute(0, 3, 1, 2)
            elif x.dtype not in (torch.uint8, torch.float16, torch.float32):
                raise TypeError("unsupported data type for torch.Tensor")
            # Resizing
            if x.shape[-2] != self.resize.size[0] or x.shape[-1] != self.resize.size[1]:  # type: ignore[union-attr]
                x = F.resize(
                    x, self.resize.size, interpolation=self.resize.interpolation, antialias=self.resize.antialias
                )
            # Data type
            if x.dtype == torch.uint8:  # type: ignore[union-attr]
                x = x.to(dtype=torch.float32).div(255).clip(0, 1)  # type: ignore[union-attr]
            else:
                x = x.to(dtype=torch.float32)  # type: ignore[union-attr]
            batches = [x]

        elif isinstance(x, list) and all(isinstance(sample, (np.ndarray, torch.Tensor)) for sample in x):
            # Sample transform (to tensor, resize)
            samples = list(multithread_exec(self.sample_transforms, x))
            # Batching
            batches = self.batch_inputs(samples)  # type: ignore[assignment]
        else:
            raise TypeError(f"invalid input type: {type(x)}")

        # Batch transforms (normalize)
        batches = list(multithread_exec(self.normalize, batches))
        
        return batches  # type: ignore[return-value]
