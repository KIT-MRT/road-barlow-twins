import os
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.rasterization import build_rasterizer


class WovenPredictionDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_dataloader_workers=8, pin_memory=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        os.environ[
            "L5KIT_DATA_FOLDER"
        ] = "/p/project/hai_mrt_pc/woven-prediction-dataset/"

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            dm = LocalDataManager(None)
            cfg = load_config_data(
                "/p/home/jusers/wagner20/juwels/road-barlow-twins/src/road_barlow_twins/data_utils/agent_motion_config.yaml"
            )
            rasterizer = build_rasterizer(cfg, dm)
            train_zarr = ChunkedDataset(
                dm.require(cfg["train_data_loader"]["key"])
            ).open()
            train_dataset = AgentDataset(cfg, train_zarr, rasterizer)

            self.train_set = train_dataset

            val_zarr = ChunkedDataset(dm.require(cfg["val_data_loader"]["key"])).open()
            val_dataset = AgentDataset(cfg, val_zarr, rasterizer)

            self.val_set = val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
