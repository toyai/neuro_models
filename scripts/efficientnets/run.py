"""
Run the training and testing script.
"""
import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from lightnings.efficientnets import EfficientNetGym
from loggers.model_logger import model_logger, model_summary_logger
from models.efficientnets import EfficientNet

log = logging.getLogger(__name__)

seed_everything(666)


@hydra.main(config_path=os.getcwd() + "/conf/model", config_name="efficientnets")
def main(cfg: DictConfig = None):
    log.info("Training Configs:\n%s", OmegaConf.to_yaml(cfg))

    network = EfficientNet(num_classes=cfg.lm.num_classes, name=cfg.name)
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

    ckpt = ModelCheckpoint("ckpt/{epoch}", prefix=cfg.name) if cfg.ckpt else False
    trainer = Trainer(**cfg.pl, logger=dl_logger, checkpoint_callback=ckpt)
    trainer.fit(gym, datamodule=dm)
    if cfg.test:
        trainer.test(datamodule=dm)


if __name__ == "__main__":
    main()
