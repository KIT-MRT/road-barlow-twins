import torch
import numpy as np
import pytorch_lightning as pl

from torch import nn
from vit_pytorch.cross_vit import CrossTransformer


class DualMotionViT(pl.LightningModule):
    def __init__(
        self,
        map_encoder,
        agent_encoder,
        map_encoder_dim=768,
        agent_encoder_dim=768,
        fusion_depth=2,
        n_traj=6,
        time_limit=50,
        lr=1e-4,
    ) -> None:
        super().__init__()
        self.n_traj = n_traj
        self.time_limit = time_limit
        self.lr = lr
        self.map_encoder = map_encoder
        self.agent_encoder = agent_encoder
        self.fusion_block = CrossTransformer(
            sm_dim=agent_encoder_dim,
            lg_dim=map_encoder_dim,
            depth=fusion_depth,
            heads=8,
            dim_head=128,
            dropout=0.2,
        )
        self.motion_head = nn.Sequential(
            nn.LayerNorm((agent_encoder_dim,), eps=1e-06, elementwise_affine=True),
            nn.Linear(
                in_features=agent_encoder_dim,
                out_features=self.n_traj * 2 * self.time_limit + self.n_traj,
            ),
        )

    def forward(self, x):
        map_data, agent_data = x[:, 0:3, ...], x[:, 3:, ...]
        map_embedding = self.map_encoder.forward_features(map_data)
        agent_embedding = self.agent_encoder.forward_features(agent_data)
        fused_embedding, _ = self.fusion_block(
            sm_tokens=agent_embedding, lg_tokens=map_embedding
        )
        fused_class_tokens = fused_embedding[:, 0, ...]
        motion_embedding = self.motion_head(fused_class_tokens)
        confidences_logits, logits = (
            motion_embedding[:, : self.n_traj],
            motion_embedding[:, self.n_traj :],
        )
        logits = logits.view(-1, self.n_traj, self.time_limit, 2)

        return confidences_logits, logits

    def _shared_step(self, batch, batch_idx):
        x, y, is_available = batch
        y = y[:, : self.time_limit, :]
        is_available = is_available[:, : self.time_limit]
        confidences_logits, logits = self.forward(x)

        loss = pytorch_neg_multi_log_likelihood_batch(
            y, logits, confidences_logits, is_available
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=190,
                    eta_min=1e-6,
                ),
                "interval": "epoch",
                "frequency": 1,
                "name": "lr",
            },
        }
    

def pytorch_neg_multi_log_likelihood_batch(gt, logits, confidences, avails):
    """Compute a negative log-likelihood for the multi-modal scenario.
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        logits (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    Src: 
        https://github.com/kbrodt/waymo-motion-prediction-2021
    """

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(
        ((gt - logits) * avails) ** 2, dim=-1
    )  # reduce coords and use availability

    with np.errstate(
        divide="ignore"
    ):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = nn.functional.log_softmax(confidences, dim=1) - 0.5 * torch.sum(
            error, dim=-1
        )  # reduce time

    # error (batch_size, num_modes)
    error = -torch.logsumexp(error, dim=-1, keepdim=True)

    return torch.mean(error)