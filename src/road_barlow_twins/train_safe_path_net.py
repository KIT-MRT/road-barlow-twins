# %cd /p/home/jusers/wagner20/juwels/road-barlow-twins/src/road_barlow_twins
import torch

from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

from l5kit.configs import load_config_data
from models.save_path_net import SafePathNet
from data_utils.dataset_modules import WovenPredictionGraphDataModule


def main():
    cfg = load_config_data("/p/home/jusers/wagner20/juwels/road-barlow-twins/src/road_barlow_twins/data_utils/save_path_net_config.yaml")
    safe_path_net = SafePathNet(cfg) 

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        precision=16,
        accelerator="gpu",
        devices=4,
        num_nodes=1,
        strategy="ddp",
        max_time={"days": 0, "hours": 15},
        default_root_dir="/p/project/hai_mrt_pc/motion-pred/graph",
        callbacks=[lr_monitor],
    )

    dm = WovenPredictionGraphDataModule(
        batch_size=512, # Test with train instead of train_full
        num_dataloader_workers=4,
        pin_memory=False, # OOM if pin memory is used
        num_train_samples=1_000_000
    )

    trainer.fit(safe_path_net, datamodule=dm)

    save_time = datetime.utcnow().replace(microsecond=0).isoformat()
    torch.save(
        safe_path_net.state_dict(),
        f"/p/project/hai_mrt_pc/motion-pred/graph/models/safe-path-net-{save_time}.pt",
    )


if __name__ == "__main__":
    main()