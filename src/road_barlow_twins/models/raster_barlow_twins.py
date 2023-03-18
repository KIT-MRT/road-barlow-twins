import torch
import pytorch_lightning as pl

from torch import nn


class BarlowTwins(pl.LightningModule):
    """Src: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/barlow-twins.html"""

    def __init__(
        self,
        encoder,
        encoder_out_dim,
        batch_size,
        lambda_coeff=5e-3,
        z_dim=128,
        learning_rate=1e-4,
        max_epochs=200,
    ):
        super().__init__()

        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(in_features=encoder_out_dim, out_features=4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=z_dim)
        )

        self.loss_fn = BarlowTwinsLoss(
            batch_size=batch_size, lambda_coeff=lambda_coeff, z_dim=z_dim
        )

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def forward(self, x):
        return self.encoder(x)

    def shared_step(self, batch):
        x1, x2 = batch

        z1 = self.projection_head(self.encoder(x1))
        z2 = self.projection_head(self.encoder(x2))

        return self.loss_fn(z1, z2)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.max_epochs,
                    eta_min=1e-6,
                ),
                "interval": "epoch",
                "frequency": 1,
                "name": "lr",
            },
        }


class BarlowTwinsLoss(nn.Module):
    """Src: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/barlow-twins.html"""

    def __init__(self, batch_size, lambda_coeff=5e-3, z_dim=128):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag
