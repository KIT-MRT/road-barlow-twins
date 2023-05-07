import os
import argparse
import torch
import timm
import pytorch_lightning as pl

from torch import nn
from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from models.raster_barlow_twins import BarlowTwins
from data_utils.dataset_modules import WaymoPredictionBarlowRasterDataModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--encoder-out-dim", type=int, required=True)
    parser.add_argument("--pretrained", action="store_true")
    # TODO: add param to load pre-training checkpoint
    parser.add_argument("--batch-size", type=int, required=False, default=128)
    parser.add_argument("--lr", type=float, required=False, default=1e-4)
    parser.add_argument("--train-hours", type=float, required=False, default=9.0)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--train-sample-limit", type=int, required=False, default=0)
    parser.add_argument("--num-nodes", type=int, required=False, default=1)
    parser.add_argument("--num-gpus", type=int, required=False, default=4)
    parser.add_argument(
        "--train-path",
        type=str,
        required=False,
        default="/p/project/hai_mrt_pc/waymo-prediction/pre-rendered/train",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        required=False,
        default="/p/project/hai_mrt_pc/waymo-prediction/pre-rendered/dev",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    csv_logger = CSVLogger(f"{args.save_dir}")

    encoder = timm.create_model(args.model, in_chans=3, pretrained=args.pretrained)
    if args.model.startswith("vit"):
        encoder.head = nn.Identity()
    else:
        encoder.fc = nn.Identity()

    bt = BarlowTwins(
        encoder=encoder,
        encoder_out_dim=args.encoder_out_dim,
        batch_size=args.batch_size,
        z_dim=2048,
        learning_rate=args.lr,
    )

    dm_barlow = WaymoPredictionBarlowRasterDataModule(
        batch_size=args.batch_size,
        num_dataloader_workers=10,
        pin_memory=True,
        train_path=args.train_path,
        val_path=args.val_path,
        val_limit=24 * 100,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        precision=16,
        accelerator="gpu",
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        max_time={"days": 0, "hours": args.train_hours},
        default_root_dir=args.save_dir,
        callbacks=[lr_monitor],
        logger=csv_logger,
        strategy="ddp",
    )

    trainer.fit(bt, datamodule=dm_barlow)

    save_time = datetime.utcnow().replace(microsecond=0).isoformat()

    torch.save(
        bt.state_dict(),
        f"{args.save_dir}/models/{args.model}-{save_time}.pt",
    )


if __name__ == "__main__":
    main()
