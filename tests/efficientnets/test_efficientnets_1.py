import os

import pytest
from hydra.experimental import compose, initialize_config_dir

from model.efficientnets.run import main as efficientnets_main


@pytest.mark.parametrize(
    ["name", "test"],
    [
        pytest.param("efficientnet-b1", True),
        pytest.param("efficientnet-b1", False),
    ],
)
def test_efficientnets(name, test):
    with initialize_config_dir(os.getcwd() + "/conf"):
        cfg = compose(
            config_name="efficientnets",
            overrides=[
                f"name={name}",
                "logger=false",
                f"test={test}",
                "pl.max_epochs=1",
                "pl.gpus=0",
                "pl.limit_train_batches=1",
                "pl.limit_val_batches=1",
                "pl.limit_test_batches=1",
            ],
        )
        efficientnets_main(cfg)
