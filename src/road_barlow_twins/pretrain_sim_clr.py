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

from data_utils.dataset_modules import WaymoPredictionBarlowRasterDataModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--encoder-out-dim", type=int, required=True)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--batch-size", type=int, required=False, default=128)
    parser.add_argument("--lr", type=float, required=False, default=1e-4)
    parser.add_argument("--train-hours", type=float, required=False, default=9.0)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--train-sample-limit", type=int, required=False, default=0)

    args = parser.parse_args()

    return args


class SimCLR(pl.LightningModule):
    """Based on: https://github.com/Spijkervet/SimCLR"""

    def __init__(self, encoder, projection_dim, n_features, batch_size, learning_rate, max_epochs=200):
        super().__init__()

        self.encoder = encoder
        self.n_features = n_features

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

        self.loss_fn = NT_XentLoss(batch_size=batch_size, temperature=0.07)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def forward(self, batch):
        x_i, x_j = batch
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        return z_i, z_j
    
    def training_step(self, batch, batch_idx):
        z_i, z_j = self.forward(batch)
        loss = self.loss_fn(z_i, z_j)
        self.log("train_loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        z_i, z_j = self.forward(batch)
        loss = self.loss_fn(z_i, z_j)
        self.log("val_loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        return loss
    
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


class NT_XentLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


def main():
    args = parse_args()
    csv_logger = CSVLogger(f"{args.save_dir}")

    encoder = timm.create_model(args.model, in_chans=3, pretrained=args.pretrained)
    if args.model.startswith("vit"):
        encoder.head = nn.Identity()
    else:
        encoder.fc = nn.Identity()

    sc = SimCLR(
        encoder=encoder,
        n_features=args.encoder_out_dim,
        batch_size=args.batch_size,
        projection_dim=2048,
        learning_rate=args.lr,
    )

    dm_barlow = WaymoPredictionBarlowRasterDataModule(
        batch_size=args.batch_size,
        num_dataloader_workers=10,
        pin_memory=True,
        train_path="/p/project/hai_mrt_pc/waymo-prediction/pre-rendered/train",
        val_path="/p/project/hai_mrt_pc/waymo-prediction/pre-rendered/dev",
        val_limit=24 * 100,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        precision=16,
        accelerator="gpu",
        devices=4,
        num_nodes=1,
        max_time={"days": 0, "hours": args.train_hours},
        default_root_dir=args.save_dir,
        callbacks=[lr_monitor],
        logger=csv_logger,
        strategy="ddp",
    )

    trainer.fit(sc, datamodule=dm_barlow)

    save_time = datetime.utcnow().replace(microsecond=0).isoformat()

    torch.save(
        sc.state_dict(),
        f"{args.save_dir}/models/{args.model}-{save_time}.pt",
    )


if __name__ == "__main__":
    main()
