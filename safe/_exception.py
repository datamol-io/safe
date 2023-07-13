class SafeDecodeError(Exception):
    """Raised when a string cannot be decoded with the given encoding."""

    pass


class SafeEncodeError(Exception):
    """Raised when a molecule cannot be encoded using SAFE."""

    pass


class SafeFragmentationError(Exception):
    """Raised when a the slicing algorithm return empty bonds."""

    pass
