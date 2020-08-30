In this folder, there will only be yaml files for each dataset which are already implemented with LightningDataModule.

In each yaml file, generally these are included.

```yaml
# @package _global_
dm: # for calling with `dm`
  _target_: datamodule.cifar10.CIFAR10DataModule # implemented LightningDataModule Class
  # arguments of implemented LightningDataModule Class
  # valid for CIFAR10DataModule
  # NOTE: can have more configs for different datasets
  train_dataloader_conf:
    batch_size: 32
    pin_memory: true
    shuffle: true
    num_workers: 4

  val_dataloader_conf:
    batch_size: 64
    pin_memory: true
    num_workers: 4
```
