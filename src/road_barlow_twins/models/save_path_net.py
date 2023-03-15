import torch
import pytorch_lightning as pl

from torch import nn, optim

from l5kit.prediction.vectorized.safepathnet_model import SafePathNetModel


class SafePathNet(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.net = SafePathNetModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            num_timesteps=cfg["model_params"]["future_num_frames"],
            weights_scaling=cfg["model_params"]["weights_scaling"],
            criterion=nn.L1Loss(reduction="none"),
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"],
            agent_num_trajectories=cfg["model_params"]["agent_num_trajectories"],
            max_num_agents=cfg["data_generation_params"]["other_agents_num"],
            cost_prob_coeff=cfg["model_params"]["cost_prob_coeff"] * 2.5,
        )

    def forward(self, data):
        return self.net(data)
    
    def training_step(self, batch, batch_idx):
        loss = self.net(batch)["loss"]
        self.log("train_loss", loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.net(batch)["loss"]
        self.log("val_loss", loss, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=50,
                    eta_min=1e-6,
                ),
                "interval": "epoch",
                "frequency": 1,
                "name": "lr",
            },
        }