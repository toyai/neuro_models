"""Run the training and testing script."""

import logging
import os

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from lightnings.efficientnets import EfficientNetGym
from loggers.model_logger import model_logger, model_summary_logger
from models.efficientnets import EfficientNet
from utils.efficientnets import Swish, compound_params, round_filters

log = logging.getLogger(__name__)

seed_everything(666)


@hydra.main(config_path=os.getcwd() + "/conf/model", config_name="efficientnets")
def main(cfg: DictConfig = None):
    log.info("Training Configs:\n%s", OmegaConf.to_yaml(cfg))

    if cfg.pretrained:
        with torch.set_grad_enabled(False):
            network = EfficientNet(
                name=cfg.name,
                num_classes=cfg.lm.num_classes,
            ).from_pretrained(name=cfg.name)

        width, _, _, dropout_p, _, _ = compound_params(cfg.name)
        final_out_channels = round_filters(1280, 8, width)

        network.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Dropout(dropout_p),
            nn.Linear(final_out_channels, cfg.lm.num_classes),
            Swish(),
        )
    else:
        network = EfficientNet(name=cfg.name, num_classes=cfg.lm.num_classes)

    gym = EfficientNetGym(network, cfg)
    dm = instantiate(cfg.dm)

    model_logger(cfg.name, network)
    model_summary_logger(cfg.name, gym)

    if cfg.logger:
        dl_logger = WandbLogger(
            name=f"{cfg.lm.optimizer}-{cfg.lm.learning_rate}",
            project=cfg.name,
        )
    else:
        dl_logger = True

    ckpt = ModelCheckpoint("ckpt/{epoch}", prefix="-" + cfg.name) if cfg.ckpt else False
    trainer = Trainer(**cfg.pl, logger=dl_logger, checkpoint_callback=ckpt)
    trainer.fit(gym, datamodule=dm)
    if cfg.test:
        trainer.test(datamodule=dm)


if __name__ == "__main__":
    main()
