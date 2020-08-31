import os

import pytest
from hydra.experimental import compose, initialize_config_dir

from model.efficientnets.run import main as efficientnets_main


@pytest.mark.parametrize(
    ["name", "dm"],
    [
        pytest.param("efficientnet-b5", "cifar10"),
        pytest.param("efficientnet-b5", "cifar100"),
    ],
)
def test_efficientnets(name, dm):
    with initialize_config_dir(os.getcwd() + "/conf"):
        cfg = compose(
            config_name="efficientnets",
            overrides=[
                f"name={name}",
                f"dm={dm}",
                "logger=false",
                "pl.max_epochs=1",
                "pl.gpus=0",
                "pl.limit_train_batches=1",
                "pl.limit_val_batches=1",
                "pl.limit_test_batches=1",
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
