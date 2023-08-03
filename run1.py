# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MNIST backbone image classifier example.

To run: python backbone_image_classifier.py --trainer.max_epochs=50
"""
# from os import path
# from typing import Optional
# import flash.image 

import torch
torch.set_default_dtype(torch.float32) 
# from pytorch_lightning import (LightningDataModule, LightningModule,
#                                cli_lightning_logo)
# from pytorch_lightning.cli import LightningCLI
# from pytorch_lightning.demos.mnist_datamodule import MNIST
# from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY, MODEL_REGISTRY
# from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE
# from torch.nn import functional as F
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms

from data.loc_data import LocDataModule

from models.models import RegreTransfL, RegreMPL
# from pytorch_lightning import loggers as pl_logger

# def cli_main():
#     # MODEL_REGISTRY.register_classes(flash.image, LightningModule)
#     cli = LightningCLI(MultiLayerPerceptrons, LocDataModule, seed_everything_default=1234, run=False)
#     cli.trainer.fit(cli.model, datamodule=cli.datamodule)
#     cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
#     predictions = cli.trainer.predict(ckpt_path="best", datamodule=cli.datamodule)
#     print(predictions[0])


if __name__ == "__main__":
    # cli_lightning_logo()
    # tb_logger = pl_logger.TensorBoardLogger('lightning_logs')
    from pytorch_lightning.cli import LightningCLI
    # from pytorch_lightning.utilities.cli import LightningCLI
    # cli = LightningCLI(MultiLayerPerceptrons, LocDataModule, save_config_callback=None, save_config_kwargs=True)
    # cli = LightningCLI(RegreTransformer, LocDataModule, save_config_callback=None, save_config_kwargs=True)
    # cli = LightningCLI(RegreTransfL, LocDataModule, save_config_callback=None, save_config_kwargs=True)
    cli = LightningCLI(RegreMPL, LocDataModule, save_config_callback=None, save_config_kwargs=True)