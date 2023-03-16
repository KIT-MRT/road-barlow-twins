import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import nn


class ConvMotionPred(pl.LightningModule):
    def __init__(self, encoder, num_past_frames, num_future_frames, lr) -> None:
        super().__init__()
        self.encoder = encoder
        # Past frames + current frames for ego agent and other agents and 3 map channels
        num_in_channels = (num_past_frames + 1) * 2 + 3
        # Predict (x, y) coordinates for each future frame
        num_out_nodes = num_future_frames * 2

        self.encoder.conv1 = nn.Conv2d(
            in_channels=num_in_channels,
            out_channels=encoder.conv1.out_channels,
            kernel_size=encoder.conv1.kernel_size,
            stride=encoder.conv1.stride,
            padding=encoder.conv1.padding,
            bias=False,
        )
        # Drop pre-trained fc layer
        self.encoder.fc = nn.Identity()
        
        self.neck = nn.Sequential(
            nn.AvgPool2d(kernel_size=3),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=2048, out_features=2048),
        )
        
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=num_out_nodes),
        )

        self.lr = lr

    def _get_preds_and_loss(self, data):
        targets = data["target_positions"]
        target_availabilities = data["target_availabilities"].unsqueeze(-1)
        
        x = data["image"]
        x = self.encoder(x)
        x = self.neck(x)
        preds = self.head(x)
        preds = preds.reshape(targets.shape)
        
        loss = F.mse_loss(preds, targets)

        # Filter out invalid predictions
        loss = loss * target_availabilities

        return preds, loss.mean()

    def training_step(self, batch, batch_idx):
        _, loss = self._get_preds_and_loss(batch)
        self.log("train_loss", loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        _, loss = self._get_preds_and_loss(batch)
        self.log("val_loss", loss, sync_dist=True)

        return loss

    def forward(self, x):
        return self._get_preds_and_loss(x)

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
