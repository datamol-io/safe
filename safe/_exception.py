class SAFEDecodeError(Exception):
    """Raised when a string cannot be decoded with the given encoding."""

    pass


class SAFEEncodeError(Exception):
    """Raised when a molecule cannot be encoded using SAFE."""

    pass


class SAFEFragmentationError(Exception):
    """Raised when a the slicing algorithm return empty bonds."""

    pass
