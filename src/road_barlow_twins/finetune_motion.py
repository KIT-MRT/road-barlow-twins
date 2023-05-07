import timm
import torch
import argparse
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

from pytorch_lightning.loggers import CSVLogger, WandbLogger

from models.dual_motion_vit import DualMotionViT
from data_utils.eval_motion_prediction import run_eval_dataframe
from data_utils.dataset_modules import WaymoPredictionDataModule


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
    parser.add_argument("--logger", type=str, required=False, default="csv")
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
    start_time = datetime.utcnow().replace(microsecond=0).isoformat()

    if args.logger == "csv":
        logger = CSVLogger(f"{args.save_dir}")
    elif args.logger == "wandb":
        logger = WandbLogger(
            project="road-barlow-twins",
            save_dir=args.save_dir,
            name=f"{args.model}-{start_time}",
            offline=True
        )

    map_encoder = timm.create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=3,
        num_classes=1_000,  # can be random
    )

    agent_encoder = timm.create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=22,
        num_classes=1_000,  # can be random
    )

    if args.checkpoint:
        map_encoder.load_state_dict(torch.load(args.checkpoint), strict=False)
    if args.checkpoint2:
        agent_encoder.load_state_dict(torch.load(args.checkpoint2), strict=False)

    motion_predictor = DualMotionViT(
        map_encoder=map_encoder, agent_encoder=agent_encoder
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
        logger=logger,
        strategy="ddp_find_unused_parameters_true",
    )

    dm = WaymoPredictionDataModule(
        batch_size=args.batch_size,
        num_dataloader_workers=12,
        pin_memory=True,
        train_path=args.train_path,
        val_path=args.val_path,
        val_limit=24 * 100,
        train_limit=args.train_sample_limit,
    )

    trainer.fit(motion_predictor, datamodule=dm)

    if trainer.is_global_zero:
        torch.save(
            motion_predictor.state_dict(),
            f"{args.save_dir}/models/{args.model}-{start_time}.pt",
        )

        pred_metrics = run_eval_dataframe(
            model=motion_predictor,
            data=args.val_path,
            prediction_horizons=[30, 50]
        )

        if args.logger == "wandb":
            logger.log_table(
                key="motion_prediction_eval",
                dataframe=pred_metrics
            )
        else:
            print(pred_metrics)


if __name__ == "__main__":
    main()
