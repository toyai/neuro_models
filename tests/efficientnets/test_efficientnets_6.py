import os

import pytest
from hydra.experimental import compose, initialize_config_dir
from hypothesis import given
from hypothesis.strategies import booleans

from scripts.efficientnets.run import main as efficientnets_main


@pytest.mark.parametrize(
    ["name", "dm", "num_classes"],
    [
        pytest.param("efficientnet-b6", "cifar10", 10),
        pytest.param("efficientnet-b6", "cifar100", 100),
    ],
)
@given(pretrained=booleans(), include_fc=booleans())
def test_efficientnets(name, dm, num_classes, pretrained, include_fc):
    with initialize_config_dir(os.getcwd() + "/conf"):
        cfg = compose(
            config_name="efficientnets",
            overrides=[
                f"name={name}",
                f"dm={dm}",
                f"pretrained={pretrained}",
                f"include_fc={include_fc}",
                f"lm.num_classes={num_classes}",
                "logger=false",
                "pl.max_epochs=1",
                "pl.gpus=0",
                "pl.limit_train_batches=5",
                "pl.limit_val_batches=5",
                "pl.limit_test_batches=5",
                "dm.train_dataloader_conf.batch_size=1",
                "dm.train_dataloader_conf.pin_memory=false",
                "dm.train_dataloader_conf.shuffle=false",
                "dm.train_dataloader_conf.num_workers=1",
                "dm.val_dataloader_conf.batch_size=1",
                "dm.val_dataloader_conf.pin_memory=false",
                "dm.val_dataloader_conf.num_workers=1",
            ],
        )
        efficientnets_main(cfg)
