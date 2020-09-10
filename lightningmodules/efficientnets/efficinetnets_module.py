import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F


class EfficientNetGym(pl.LightningModule):
    """EfficientNet is working out here. 🏋️‍♀️"""

    def __init__(self, model: nn.Module, cfg: DictConfig):
        super().__init__()
        self.model = model
        self.hparams = cfg

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return instantiate(self.hparams.optim, **{"params": self.parameters()})

    def _topk(self, pred, target, topk=(1,)):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L411
        """
        Computes the accuracy over the k top predictions for the specified values of k.
        """
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = pred.topk(maxk, 1, True, True)  # pred.size - (batch_size, maxk)
            pred = pred.t()  # (maxk, batch_size)
            # .view makes target size (1, batch_size)
            # expand_as results in shape of (maxk, batch_size)
            # now we are comparing pred and target element-wise
            # if there is `True` at 1st row of comparing pred and target,
            # this is for top1 acc.
            # if there is at 5th row, this is for top5 acc.
            # if there is at kth row, this is for topk acc.
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
                res.append(correct_k / batch_size)

            return res

    def _step(self, batch, batch_idx):
        img, target = batch
        pred = self(img)
        loss = F.cross_entropy(pred, target)
        top1, top5 = self._topk(pred, target, (1, 5))
        return loss, top1, top5

    def training_step(self, batch, batch_idx):
        loss, top1, top5 = self._step(batch, batch_idx)
        logs = {
            "step_train_loss": loss,
            "step_train_top1": top1,
            "step_train_top5": top5,
        }

        return {
            "loss": loss,
            "top1": top1,
            "top5": top5,
            "log": logs,
        }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_top1 = torch.stack([x["top1"] for x in outputs]).mean()
        avg_top5 = torch.stack([x["top5"] for x in outputs]).mean()

        logs = {
            "epoch_train_loss": avg_loss,
            "epoch_train_top1": avg_top1,
            "epoch_train_top5": avg_top5,
        }

        return {"log": logs}

    def validation_step(self, batch, batch_idx):
        loss, top1, top5 = self._step(batch, batch_idx)
        # no log at valiation step
        return {
            "val_loss": loss,
            "val_top1": top1,
            "val_top5": top5,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_top1 = torch.stack([x["val_top1"] for x in outputs]).mean()
        avg_top5 = torch.stack([x["val_top5"] for x in outputs]).mean()
        logs = {
            "val_loss": avg_loss,
            "val_top1": avg_top1,
            "val_top5": avg_top5,
        }

        return {"log": logs}

    def test_step(self, batch, batch_idx):
        loss, top1, top5 = self._step(batch, batch_idx)
        # no log at test step
        return {
            "test_loss": loss,
            "test_top1": top1,
            "test_top5": top5,
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_top1 = torch.stack([x["test_top1"] for x in outputs]).mean()
        avg_top5 = torch.stack([x["test_top5"] for x in outputs]).mean()
        logs = {
            "test_loss": avg_loss,
            "test_top1": avg_top1,
            "test_top5": avg_top5,
        }

        return {"log": logs}
