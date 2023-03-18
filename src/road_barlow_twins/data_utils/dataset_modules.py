import os
import torch
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Subset, Dataset

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoAgentDatasetVectorized
from l5kit.rasterization import build_rasterizer
from l5kit.vectorization.vectorizer_builder import build_vectorizer

from raster_barlow_twins_transform import BarlowTwinsTransform


class WaymoBarlowRasterLoader(Dataset):
    def __init__(self, directory, limit=0, return_vector=False, is_test=False):
        files = os.listdir(directory)
        self.files = [os.path.join(directory, f) for f in files if f.endswith(".npz")]

        if limit > 0:
            self.files = self.files[:limit]
        else:
            self.files = sorted(self.files)

        self.return_vector = return_vector
        self.is_test = is_test
        self.transform = BarlowTwinsTransform(
            train=True, input_height=224, gaussian_blur=False, jitter_strength=0.5
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        data = np.load(filename, allow_pickle=True)

        raster = data["raster"].astype("float32")
        raster = raster.transpose(2, 1, 0) / 255

        if self.return_vector:
            return raster, data["vector_data"]

        return self.transform(raster[0:3])


class WaymoPredictionBarlowRasterDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        num_dataloader_workers=8,
        pin_memory=True,
        train_path="",
        val_path="",
        val_limit=200,
        train_limit=0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.pin_memory = pin_memory
        self.train_path = train_path
        self.val_path = val_path
        self.val_limit = val_limit
        self.train_limit = train_limit

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_set = WaymoBarlowRasterLoader(
                self.train_path, limit=self.train_limit
            )
            self.val_set = WaymoBarlowRasterLoader(self.val_path, limit=self.val_limit)

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


class WovenPredictionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        num_dataloader_workers=8,
        pin_memory=True,
        num_train_samples=500_000,
        step_size_slicing=10,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.pin_memory = pin_memory
        self.num_train_samples = num_train_samples
        self.step_size_slicing = step_size_slicing

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

            self.train_set = Subset(
                train_dataset,
                torch.arange(
                    start=0,
                    end=self.num_train_samples * self.step_size_slicing,
                    step=self.step_size_slicing,
                ),
            )
            self.val_set = Subset(
                train_dataset,
                torch.arange(
                    start=self.num_train_samples * self.step_size_slicing,
                    end=self.num_train_samples * self.step_size_slicing + 10_000,
                    step=1,
                ),
            )

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


class WovenPredictionGraphDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        num_dataloader_workers=8,
        pin_memory=True,
        num_train_samples=500_000,
        step_size_slicing=10,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.pin_memory = pin_memory
        self.num_train_samples = num_train_samples
        self.step_size_slicing = step_size_slicing

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
                "/p/home/jusers/wagner20/juwels/road-barlow-twins/src/road_barlow_twins/data_utils/save_path_net_config.yaml"
            )
            vectorizer = build_vectorizer(cfg, dm)
            train_zarr = ChunkedDataset(
                dm.require(cfg["train_data_loader"]["key"])
            ).open()
            train_dataset = EgoAgentDatasetVectorized(cfg, train_zarr, vectorizer)

            self.train_set = Subset(
                train_dataset,
                torch.arange(
                    start=0,
                    end=self.num_train_samples * self.step_size_slicing,
                    step=self.step_size_slicing,
                ),
            )
            self.val_set = Subset(
                train_dataset,
                torch.arange(
                    start=self.num_train_samples * self.step_size_slicing,
                    end=self.num_train_samples * self.step_size_slicing + 10_000,
                    step=1,
                ),
            )

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
