import pytest

import safe


@pytest.mark.parametrize("highlight_mode", ["lasso", "fill", "color", None])
@pytest.mark.parametrize("use_svg", [True, False])
def test_safe_visualization_modes(highlight_mode, use_svg):
    encoded = safe.encode("CCOc1ccccc1", canonical=True)

    image = safe.to_image(encoded, highlight_mode=highlight_mode, use_svg=use_svg)

    assert image is not None


def test_safe_visualization_rejects_unknown_mode():
    with pytest.raises(ValueError, match="highlight_mode"):
        safe.to_image("CC", highlight_mode="unknown")


def test_safe_visualization_accepts_explicit_fragment_sequence():
    encoded = safe.encode("CCOc1ccccc1", canonical=True)

    image = safe.to_image(encoded, fragments=["CCO", "c1ccccc1"], use_svg=True)

    assert image is not None
