def test_import():
    import importlib

    assert importlib.import_module("safe") is not None
