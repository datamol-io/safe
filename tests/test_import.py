def test_import():
    import importlib
    import subprocess
    import sys

    assert importlib.import_module("safe") is not None
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import safe, sys; "
            "assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules; "
            "assert safe.split('C%(100).N%99') == ['C', '%(100)', '.', 'N', '%99']; "
            "print(safe.__version__)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip()

    subprocess.run(
        [
            sys.executable,
            "-c",
            "from safe import *; import sys; "
            "assert 'torch' not in sys.modules; "
            "assert callable(encode) and split('C.C') == ['C', '.', 'C']",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
