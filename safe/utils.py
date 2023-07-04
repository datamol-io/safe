from contextlib import contextmanager


@contextmanager
def attr_as(obj, field, value):
    """Temporary replace the value of an object
    Args:
        obj: object to temporary patch
        field: name of the key to change
        value: value of key to be temporary changed
    """
    old_value = getattr(obj, field, None)
    setattr(obj, field, value)
    yield
    setattr(obj, field, old_value)
