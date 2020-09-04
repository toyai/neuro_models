import pytest
from torch import nn

from models.efficientnets import EfficientNet
from utils.efficientnets import load_weights


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("efficientnet-b0"),
        pytest.param("efficientnet-b1"),
        pytest.param("efficientnet-b2"),
        pytest.param("efficientnet-b3"),
        pytest.param("efficientnet-b4"),
        pytest.param("efficientnet-b5"),
        pytest.param("efficientnet-b6"),
        pytest.param("efficientnet-b7"),
    ],
)
def test_invalid_load_weights(name):
    model = EfficientNet(name, 1000)
    model.classifier.add_module("linear", nn.Linear(10, 1))
    with pytest.raises(RuntimeError):
        load_weights(model, name)
