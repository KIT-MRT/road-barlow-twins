import torch
import argparse
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

from pytorch_lightning.loggers import CSVLogger, WandbLogger

from models.road_env_description import REDMotionPredictor
from data_utils.dataset_modules import WaymoRoadEnvGraphDataModule

from data_utils.eval_motion_prediction import run_eval_dataframe, run_waymo_eval_per_class


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=False, default='')
    parser.add_argument("--checkpoint-red-encoder", type=str, required=False, default='')
    parser.add_argument("--checkpoint-ego-encoder", type=str, required=False, default='')
    parser.add_argument("--use-auxiliary-rbt-loss", action="store_true")
    parser.add_argument("--auxiliary-rbt-loss-weight", type=float, required=False, default=0.3)
    parser.add_argument("--prediction-subsampling-rate", type=int, required=False, default=1)
    parser.add_argument("--batch-size", type=int, required=False, default=128)
    parser.add_argument("--lr", type=float, required=False, default=1e-3)
    parser.add_argument("--train-hours", type=float, required=False, default=9.0)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--train-sample-limit", type=int, required=False, default=0)
    parser.add_argument("--num-nodes", type=int, required=False, default=1)
    parser.add_argument("--num-gpus", type=int, required=False, default=4)
    parser.add_argument("--prediction-horizon", type=int, required=False, default=50)
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
        ),
        WandbLogger(
            project="road-barlow-twins",
            save_dir=args.save_dir,
            name=f"{args.run_prefix}-{args.model}-{start_time}",
            offline=True,
        ),
    ]

    model = REDMotionPredictor(
        dim_road_env_encoder=256,
        dim_road_env_attn_window=16,
        dim_ego_trajectory_encoder=128,
        num_trajectory_proposals=6,
        prediction_horizon=args.prediction_horizon,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        auxiliary_rbt_loss=args.use_auxiliary_rbt_loss,
        auxiliary_loss_weight=args.auxiliary_rbt_loss_weight,
        prediction_subsampling_rate=args.prediction_subsampling_rate,
    )

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
    
    if args.checkpoint_red_encoder:
        model.load_state_dict(torch.load(args.checkpoint_red_encoder), strict=False)
    
    if args.checkpoint_ego_encoder:
        model.load_state_dict(torch.load(args.checkpoint_ego_encoder), strict=False)

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

        if args.prediction_horizon > 50:
            prediction_horizons = [30, 50, args.prediction_horizon]
        else:
            prediction_horizons = [30, 50]

        pred_metrics, pred_metrics_per_class = run_waymo_eval_per_class(
            model=model,
            data=args.val_path,
            prediction_horizons=prediction_horizons,
            red_model=True,
            prediction_subsampling_rate=args.prediction_subsampling_rate,
        )
        loggers[1].log_table(
            key="motion_prediction_eval",
            dataframe=pred_metrics
        )
        loggers[1].log_table(
            key="motion_prediction_eval_per_class",
            dataframe=pred_metrics_per_class
        )


if __name__ == "__main__":
    main()