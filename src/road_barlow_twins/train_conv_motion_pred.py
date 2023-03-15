# %cd /p/home/jusers/wagner20/juwels/road-barlow-twins/src/road_barlow_twins
import torch

from datetime import datetime
from pytorch_lightning import Trainer
from torchvision.models.resnet import resnet50
from pytorch_lightning.callbacks import LearningRateMonitor

from models.conv_motion_pred import ConvMotionPred
from data_utils.dataset_modules import WovenPredictionDataModule


def main():
    encoder = resnet50(pretrained=False)
    conv_motion_pred = ConvMotionPred(
        encoder=encoder,
        num_past_frames=19,
        num_future_frames=30,
        lr=1e-3,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        precision=16,
        accelerator="gpu",
        devices=4,
        num_nodes=1,
        strategy="ddp",
        max_time={"days": 0, "hours": 15},
        default_root_dir="/p/project/hai_mrt_pc/motion-pred/conv",
        callbacks=[lr_monitor],
    )

    dm = WovenPredictionDataModule(
        batch_size=512, # Test with train instead of train_full
        num_dataloader_workers=4,
        pin_memory=False, # OOM if pin memory is used
        num_train_samples=1_000_000
    )

    trainer.fit(conv_motion_pred, datamodule=dm)

    save_time = datetime.utcnow().replace(microsecond=0).isoformat()
    torch.save(
        conv_motion_pred.state_dict(),
        f"/p/project/hai_mrt_pc/motion-pred/conv/models/r50-{save_time}.pt",
    )


if __name__ == "__main__":
    main()
