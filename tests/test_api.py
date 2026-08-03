from single_prediction.api import confidence_label


def test_confidence_label():
    assert confidence_label(0.1) == "high"
    assert confidence_label(0.3) == "medium"
    assert confidence_label(0.5) == "low"
    assert confidence_label(0.9) == "high"
