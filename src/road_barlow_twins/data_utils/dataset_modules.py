import os
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

from .raster_barlow_twins_transform import BarlowTwinsTransform
from .road_env_graph_utils import (
    RoadEnvGraphAugmentations,
    waymo_vectors_to_road_env_graph,
)


class WaymoRoadEnvGraphDataModule(pl.LightningDataModule):
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
            self.train_set = WaymoRoadEnvGraphDataset(
                self.train_path, limit=self.train_limit
            )
            self.val_set = WaymoRoadEnvGraphDataset(self.val_path, limit=self.val_limit)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            drop_last=True,
        )


class WaymoRoadEnvGraphDataset(Dataset):
    def __init__(self, directory, limit=0, is_test=False, augment=False, max_len=1200):
        files = os.listdir(directory)
        self.files = [os.path.join(directory, f) for f in files if f.endswith(".npz")]

        if limit > 0:
            self.files = self.files[:limit]
        else:
            self.files = sorted(self.files)

        self.is_test = is_test
        self.max_len = max_len
        self.transform = RoadEnvGraphAugmentations()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        data = np.load(filename, allow_pickle=True)

        vectors = data["vector_data"].astype("float32")
        road_graph = waymo_vectors_to_road_env_graph(
            vectors, max_dist=50, lane_sampling_rate=3, agent_radius=25
        )
        sample_a, sample_b = self.transform(road_graph)
        sample_len = sample_a.size(dim=0)

        pad_len = 0
        if sample_len < self.max_len:
            pad_len = self.max_len - sample_len

        return {
            "sample_a": {
                "idx_src_tokens": (
                    F.pad(sample_a[:, 2], pad=(0, pad_len), value=10)
                ).int(),  # [pad] token at idx 10
                "pos_src_tokens": (
                    F.pad(sample_a[:, 0:2], pad=(0, 0, 0, pad_len), value=0)
                ).float(),
            },
            "sample_b": {
                "idx_src_tokens": (
                    F.pad(sample_b[:, 2], pad=(0, pad_len), value=10)
                ).int(),
                "pos_src_tokens": (
                    F.pad(sample_b[:, 0:2], pad=(0, 0, 0, pad_len), value=0)
                ).float(),
            },
            "src_attn_mask": (
                F.pad(torch.ones(sample_len), pad=(0, pad_len), value=0)
            ).bool(),
        }


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
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            drop_last=True,
        )


class WaymoLoader(Dataset):
    def __init__(self, directory, limit=0, return_vector=False, is_test=False):
        files = os.listdir(directory)
        self.files = [os.path.join(directory, f) for f in files if f.endswith(".npz")]

        if limit > 0:
            self.files = self.files[:limit]
        else:
            self.files = sorted(self.files)

        self.return_vector = return_vector
        self.is_test = is_test

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        data = np.load(filename, allow_pickle=True)

        raster = data["raster"].astype("float32")
        raster = raster.transpose(2, 1, 0) / 255

        if self.is_test:
            center = data["shift"]
            yaw = data["yaw"]
            agent_id = data["object_id"]
            scenario_id = data["scenario_id"]

            return (
                raster,
                center,
                yaw,
                agent_id,
                str(scenario_id),
                data["_gt_marginal"],
                data["gt_marginal"],
            )

        trajectory = data["gt_marginal"]

        is_available = data["future_val_marginal"]

        if self.return_vector:
            return raster, trajectory, is_available, data["vector_data"]

        return raster, trajectory, is_available


class WaymoPredictionDataModule(pl.LightningDataModule):
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
            self.train_set = WaymoLoader(self.train_path, limit=self.train_limit)
            self.val_set = WaymoLoader(self.val_path, limit=self.val_limit)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            drop_last=True,
        )
