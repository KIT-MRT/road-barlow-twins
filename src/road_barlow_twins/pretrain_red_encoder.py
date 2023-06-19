import torch
import argparse
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

from pytorch_lightning.loggers import CSVLogger, WandbLogger

from models.road_env_description import REDEncoder
from data_utils.dataset_modules import WaymoRoadEnvGraphDataModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--checkpoint", type=str, required=False, default="")
    parser.add_argument("--checkpoint2", type=str, required=False, default="")
    parser.add_argument("--batch-size", type=int, required=False, default=128)
    parser.add_argument("--lr", type=float, required=False, default=1e-3)
    parser.add_argument("--train-hours", type=float, required=False, default=9.0)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--train-sample-limit", type=int, required=False, default=0)
    parser.add_argument("--num-nodes", type=int, required=False, default=1)
    parser.add_argument("--num-gpus", type=int, required=False, default=4)
    parser.add_argument("--time-limit", type=int, required=False, default=80)
    parser.add_argument("--run-prefix", type=str, required=False, default="")
    parser.add_argument(
        "--train-path",
        type=str,
        required=False,
        default="/p/project/hai_mrt_pc/waymo-open-motion-dataset/motion-cnn/train-300k",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        required=False,
        default="/p/project/hai_mrt_pc/waymo-open-motion-dataset/motion-cnn/val",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    start_time = datetime.utcnow().replace(microsecond=0).isoformat()

    loggers = [
        CSVLogger(
            save_dir=f"{args.save_dir}",
            version=f"{args.model}-{start_time}",
            prefix=args.run_prefix,
        ),
        WandbLogger(
            project="road-barlow-twins",
            save_dir=args.save_dir,
            name=f"{args.run_prefix}-{args.model}-{start_time}",
            offline=True,
        ),
    ]

    model = REDEncoder(
        dim_model=256,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dim_attn_window_encoder=16,
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
        logger=loggers,
        strategy="ddp_find_unused_parameters_true",
    )

    dm = WaymoRoadEnvGraphDataModule(
        batch_size=args.batch_size,
        num_dataloader_workers=12,
        pin_memory=True,
        train_path=args.train_path,
        val_path=args.val_path,
        val_limit=24 * 1000,
        train_limit=args.train_sample_limit,
    )

    trainer.fit(model, datamodule=dm)

    if trainer.is_global_zero:
        torch.save(
            model.state_dict(),
            f"{args.save_dir}/models/{args.run_prefix}-{args.model}-{start_time}.pt",
        )

if __name__ == "__main__":
    main()
