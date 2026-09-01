"""Public SAFE API with optional model features loaded on demand."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

from . import utils
from ._exception import SAFEDecodeError, SAFEEncodeError, SAFEFragmentationError
from ._tokenizer_utils import SAFESplitter, split
from .converter import SAFEConverter, decode, encode

try:
    __version__ = version("safe-mol")
except PackageNotFoundError:  # pragma: no cover - source tree without an installation
    __version__ = "unknown"


_LAZY_IMPORTS = {
    "SAFEDesign": (".sample", "SAFEDesign", "model"),
    "SAFETokenizer": (".tokenizer", "SAFETokenizer", "model"),
    "to_image": (".viz", "to_image", "viz"),
    "upload_to_wandb": (".io", "upload_to_wandb", "wandb"),
}


def __getattr__(name):
    if name == "trainer":
        try:
            value = import_module(".trainer", __name__)
        except ModuleNotFoundError as error:
            raise ImportError(
                'SAFE training requires: python -m pip install "safe-mol[train]"'
            ) from error
        globals()[name] = value
        return value

    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute, extra = target
    try:
        value = getattr(import_module(module_name, __name__), attribute)
    except ModuleNotFoundError as error:
        raise ImportError(
            f'SAFE {name} support requires: python -m pip install "safe-mol[{extra}]"'
        ) from error
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_IMPORTS) | {"trainer"})


__all__ = [
    "SAFEConverter",
    "SAFEDecodeError",
    "SAFEDesign",
    "SAFEEncodeError",
    "SAFEFragmentationError",
    "SAFESplitter",
    "SAFETokenizer",
    "decode",
    "encode",
    "split",
    "to_image",
    "trainer",
    "upload_to_wandb",
    "utils",
]
