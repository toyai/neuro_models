import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from lightnings.densenets import DenseNetLightning
from models.densenets import DenseNet

log = logging.getLogger(__name__)

seed_everything(666)

@hydra.main(config_path=os.getcwd() + "/conf", config_name="densenets")
def main(cfg: DictConfig = None):
    log.info("Training Configs:\n%s", OmegaConf.to_yaml(cfg))

    network = DenseNet(name=cfg.name, num_classes=cfg.num_classes, bn_size=cfg.bn_size, drop_rate=cfg.drop_rate)
    gym = DenseNetLightning(network, cfg)
    transforms = T.Compose(
        [
            T.Resize(size=(img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    dm = instantiate(
        cfg.dm,
        **{"train_transforms_conf": transforms, "test_transforms_conf": transforms},
    )

    ckpt = ModelCheckpoint("ckpt/{epoch}", prefix=cfg.name) if cfg.ckpt else False
    trainer = Trainer(**cfg.pl, checkpoint_callback=ckpt)
    trainer.fit(gym, datamodule=dm)
    if cfg.test:
        trainer.test(datamodule=dm)


if __name__ == "__main__":
    main()
