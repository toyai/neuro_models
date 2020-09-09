"""Run the training and testing script."""

import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from lightningmodules.efficientnets import EfficientNetGym
from models.efficientnets import EfficientNet
from utils.efficientnets import compound_params, round_filters

log = logging.getLogger(__name__)

seed_everything(666)


@hydra.main(config_path=os.getcwd() + "/conf", config_name="efficientnets")
def main(cfg: DictConfig = None):
    log.info("Training Configs:\n%s", OmegaConf.to_yaml(cfg))

    width, _, img_size, dropout_p, _, _ = compound_params(cfg.name)

    if cfg.pretrained:
        network = EfficientNet(
            name=cfg.name,
            num_classes=cfg.num_classes,
        ).from_pretrained(name=cfg.name)
        for params in network.parameters():
            params.requires_grad = False

        final_out_channels = round_filters(1280, 8, width)

        network.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Dropout(dropout_p),
            nn.Linear(final_out_channels, cfg.num_classes),
        )
    else:
        network = EfficientNet(name=cfg.name, num_classes=cfg.num_classes)

    gym = EfficientNetGym(network, cfg)
    dm = instantiate(cfg.dm)

    with open(f"{cfg.name}.md", "w") as f:
        f.write(f"## {cfg.name}\n```py\n")
        f.write(str(network))
        f.write("\n```")

    with open(f"{cfg.name}-summary.md", "w") as f:
        f.write(f"## {cfg.name}-summary\n```py\n")
        f.write(str(ModelSummary(gym, "full")))
        f.write("\n```")

    if cfg.logger:
        logger_ = WandbLogger(
            name=f"{cfg.optim}",
            project=cfg.name,
        )
        logger_.watch(network, "all")
    else:
        logger_ = True

    ckpt = ModelCheckpoint("ckpt/{epoch}", prefix="-" + cfg.name) if cfg.ckpt else False
    trainer = Trainer(**cfg.pl, logger=logger_, checkpoint_callback=ckpt)
    trainer.fit(gym, datamodule=dm)
    if cfg.test:
        trainer.test(datamodule=dm)


if __name__ == "__main__":
    main()
