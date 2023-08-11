import torch
torch.set_default_dtype(torch.float32) 

from data.loc_data import LocDataModule

from models.models import RegreTransfL, RegreMPL, RegreMPL1

if __name__ == "__main__":
    # cli_lightning_logo()
    # tb_logger = pl_logger.TensorBoardLogger('lightning_logs')
    from pytorch_lightning.cli import LightningCLI
    # from pytorch_lightning.utilities.cli import LightningCLI
    cli = LightningCLI(RegreMPL1, LocDataModule, save_config_callback=None, save_config_kwargs=True)