# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
from typing import Any
import os
import re
import urllib
import urllib.error
import urllib.request
from pathlib import Path
from tqdm.auto import tqdm
import torch
from torch import nn

USER_AGENT = "mindee/doctr"

__all__ = [
    "load_pretrained_params",
    "conv_sequence_pt",
    "set_device_and_dtype",
    "export_model_to_onnx",
    "_copy_tensor",
    "_bf16_to_float32",
    "_CompiledModule",
]

# torch compiled model type
_CompiledModule = torch._dynamo.eval_frame.OptimizedModule

def download_from_url(
    url: str,
    file_name: str | None = None,
    hash_prefix: str | None = None,
    cache_dir: str | None = None,
    cache_subdir: str | None = None,
) -> Path:
    """Download a file using its URL

    >>> from doctr.models import download_from_url
    >>> download_from_url("https://yoursource.com/yourcheckpoint-yourhash.zip")

    Args:
        url: the URL of the file to download
        file_name: optional name of the file once downloaded
        hash_prefix: optional expected SHA256 hash of the file
        cache_dir: cache directory
        cache_subdir: subfolder to use in the cache

    Returns:
        the location of the downloaded file

    Note:
        You can change cache directory location by using `DOCTR_CACHE_DIR` environment variable.
    """
    if not isinstance(file_name, str):
        file_name = url.rpartition("/")[-1].split("&")[0]

    cache_dir = (
        str(os.environ.get("DOCTR_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "doctr")))
        if cache_dir is None
        else cache_dir
    )

    # Check hash in file name
    if hash_prefix is None:
        r = HASH_REGEX.search(file_name)
        hash_prefix = r.group(1) if r else None

    folder_path = Path(cache_dir) if cache_subdir is None else Path(cache_dir, cache_subdir)
    file_path = folder_path.joinpath(file_name)
    # Check file existence
    if file_path.is_file() and (hash_prefix is None or _check_integrity(file_path, hash_prefix)):
        logging.info(f"Using downloaded & verified file: {file_path}")
        return file_path

    try:
        # Create folder hierarchy
        folder_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        error_message = f"Failed creating cache directory at {folder_path}"
        if os.environ.get("DOCTR_CACHE_DIR", ""):
            error_message += " using path from 'DOCTR_CACHE_DIR' environment variable."
        else:
            error_message += (
                ". You can change default cache directory using 'DOCTR_CACHE_DIR' environment variable if needed."
            )
        logging.error(error_message)
        raise
    # Download the file
    try:
        print(f"Downloading {url} to {file_path}")
        _urlretrieve(url, file_path)
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print(f"Failed download. Trying https -> http instead. Downloading {url} to {file_path}")
            _urlretrieve(url, file_path)
        else:
            raise e

    # Remove corrupted files
    if isinstance(hash_prefix, str) and not _check_integrity(file_path, hash_prefix):
        # Remove file
        os.remove(file_path)
        raise ValueError(f"corrupted download, the hash of {url} does not match its expected value")

    return file_path


def _copy_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.clone().detach()


def _bf16_to_float32(x: torch.Tensor) -> torch.Tensor:
    # bfloat16 is not supported in .numpy(): torch/csrc/utils/tensor_numpy.cpp:aten_to_numpy_dtype
    return x.float() if x.dtype == torch.bfloat16 else x

def _urlretrieve(url: str, filename: Path | str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)

def load_pretrained_params(
    model: nn.Module,
    url: str | None = None,
    hash_prefix: str | None = None,
    ignore_keys: list[str] | None = None,
    cache_dir: str | None = "models/Handwritten",
    **kwargs: Any,
) -> None:
    """Load a set of parameters onto a model

    >>> from doctr.models import load_pretrained_params
    >>> load_pretrained_params(model, "https://yoursource.com/yourcheckpoint-yourhash.zip")

    Args:
        model: the PyTorch model to be loaded
        url: URL of the zipped set of parameters
        hash_prefix: first characters of SHA256 expected hash
        ignore_keys: list of weights to be ignored from the state_dict
        **kwargs: additional arguments to be passed to `doctr.utils.data.download_from_url`
    """
    model_name = url.split("/")[-1].split(".")[0].replace("-", "_")
    file_path = os.path.join(cache_dir, model_name + ".pt")
    if cache_dir is not None and os.path.exists(file_path):
        state_dict = torch.load(file_path, map_location="cpu")

    elif url is None:
        logging.warning("Invalid model URL, using default initialization.")
        return

    elif cache_dir is not None:
        # Download the file
        try:
            print(f"Downloading {url} to {file_path}")
            _urlretrieve(url, file_path)
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print(f"Failed download. Trying https -> http instead. Downloading {url} to {file_path}")
                _urlretrieve(url, file_path)
            else:
                raise e

        # # Remove corrupted files
        # if isinstance(hash_prefix, str) and not _check_integrity(file_path, hash_prefix):
        #     # Remove file
        #     os.remove(file_path)
        #     raise ValueError(f"corrupted download, the hash of {url} does not match its expected value")
        
        # Read state_dict
        state_dict = torch.load(file_path, map_location="cpu")
    else:
        file_path = download_from_url(url, hash_prefix=hash_prefix, cache_dir=cache_dir, **kwargs)
        # Read state_dict
        state_dict = torch.load(file_path, map_location="cpu")
        torch.save(state_dict, os.path.join(cache_dir, model_name + ".pt"))

    # Remove weights from the state_dict
    if ignore_keys is not None and len(ignore_keys) > 0:
        for key in ignore_keys:
            state_dict.pop(key)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if set(missing_keys) != set(ignore_keys) or len(unexpected_keys) > 0:
            raise ValueError("unable to load state_dict, due to non-matching keys.")
    else:
        # Load weights
        model.load_state_dict(state_dict)


def conv_sequence_pt(
    in_channels: int,
    out_channels: int,
    relu: bool = False,
    bn: bool = False,
    **kwargs: Any,
) -> list[nn.Module]:
    """Builds a convolutional-based layer sequence

    >>> from torch.nn import Sequential
    >>> from doctr.models import conv_sequence
    >>> module = Sequential(conv_sequence(3, 32, True, True, kernel_size=3))

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        relu: whether ReLU should be used
        bn: should a batch normalization layer be added
        **kwargs: additional arguments to be passed to the convolutional layer

    Returns:
        list of layers
    """
    # No bias before Batch norm
    kwargs["bias"] = kwargs.get("bias", not bn)
    # Add activation directly to the conv if there is no BN
    conv_seq: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, **kwargs)]

    if bn:
        conv_seq.append(nn.BatchNorm2d(out_channels))

    if relu:
        conv_seq.append(nn.ReLU(inplace=True))

    return conv_seq


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


def export_model_to_onnx(model: nn.Module, model_name: str, dummy_input: torch.Tensor, **kwargs: Any) -> str:
    """Export model to ONNX format.

    >>> import torch
    >>> from doctr.models.classification import resnet18
    >>> from doctr.models.utils import export_model_to_onnx
    >>> model = resnet18(pretrained=True)
    >>> export_model_to_onnx(model, "my_model", dummy_input=torch.randn(1, 3, 32, 32))

    Args:
        model: the PyTorch model to be exported
        model_name: the name for the exported model
        dummy_input: the dummy input to the model
        kwargs: additional arguments to be passed to torch.onnx.export

    Returns:
        the path to the exported model
    """
    torch.onnx.export(
        model,
        dummy_input,
        f"{model_name}.onnx",
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        export_params=True,
        verbose=False,
        **kwargs,
    )
    logging.info(f"Model exported to {model_name}.onnx")
    return f"{model_name}.onnx"

# if __name__ == "__main__":
    # import os
    # print(os.getcwd())
    # x = torch.randn(10, 10)
    # torch.save(x, "models/x.pt")