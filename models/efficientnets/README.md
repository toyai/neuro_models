## EfficientNet

This is the PyTorch Lightning Implementation of EfficientNet, includes EfficientNet B0-7.

Help script:

```bash
python -m model.efficientnets.run -h
```

Help output:

```bash
run is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

dm: cifar10, cifar100  <-- available datasets

== Config ==
Override anything in the config (foo.bar=value)

name: efficientnet-b0
logger: true
ckpt: true
test: true
lm:
  optimizer: Adam
  learning_rate: 0.001
  weight_decay: 0.05
  num_classes: 10
pl:
  max_epochs: 10
  gpus: -1
  fast_dev_run: false
  overfit_batches: 0
dm:
  _target_: datamodule.cifar10.CIFAR10DataModule
  train_dataloader_conf:
    batch_size: 32
    pin_memory: true
    shuffle: true
    num_workers: 4
  val_dataloader_conf:
    batch_size: 64
    pin_memory: true
    num_workers: 4

Powered by Hydra (https://hydra.cc)
Use --hydra-help to view Hydra specific help
```

Run script:

```bash
python -m model.efficientnets.run pl.max_epochs=20
```
