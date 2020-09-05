import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.metrics.functional import f1_score
from torch import nn
from torch.nn import functional as F


class EfficientNetGym(pl.LightningModule):
    """EfficientNet is working out here. 🏋️‍♀️"""

    def __init__(self, model: nn.Module, cfg: DictConfig):
        super().__init__()
        self.model = model
        self.example_input_array = torch.rand(1, 3, 224, 224)
        self.hparams = cfg

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return instantiate(self.hparams.optim)
        # optim = getattr(torch.optim, self.hparams.lm.optimizer)
        # return optim(
        #     self.parameters(),
        #     lr=self.hparams.lm.learning_rate,
        #     weight_decay=self.hparams.lm.weight_decay,
        # )

    def training_step(self, batch, batch_idx):
        img, target = batch
        pred = self(img)
        loss = F.cross_entropy(pred, target)
        acc = f1_score(F.log_softmax(pred, dim=1), target)
        logs = {"step_train_loss": loss, "step_train_acc": acc}

        return {"loss": loss, "acc": acc, "log": logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["acc"] for x in outputs]).mean()
        logs = {"epoch_train_loss": avg_loss, "epoch_train_acc": avg_acc}

        return {"log": logs}

    def validation_step(self, batch, batch_idx):
        img, target = batch
        pred = self(img)
        loss = F.cross_entropy(pred, target)
        acc = f1_score(F.log_softmax(pred, dim=1), target)
        # no log at valiation step
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss, "val_acc": avg_acc}

        return {"log": logs}

    def test_step(self, batch, batch_idx):
        img, target = batch
        pred = self(img)
        loss = F.cross_entropy(pred, target)
        acc = f1_score(F.log_softmax(pred, dim=1), target)
        # no log at test step
        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        logs = {"test_loss": avg_loss, "test_acc": avg_acc}

        return {"log": logs}
