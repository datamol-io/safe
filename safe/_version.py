try:
    from importlib.metadata import version
    from importlib.metadata import PackageNotFoundError
except ModuleNotFoundError:
    # Try backported to PY<38 `importlib_metadata`.
    from importlib_metadata import version
    from importlib_metadata import PackageNotFoundError


try:
    __version__ = "0.0.2"
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.2"
