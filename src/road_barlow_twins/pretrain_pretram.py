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

from data_utils.dataset_modules import WaymoPredictionDataModule
from models.dual_motion_vit import DualMotionViT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--encoder-out-dim", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, required=False, default="")
    parser.add_argument("--batch-size", type=int, required=False, default=128)
    parser.add_argument("--lr", type=float, required=False, default=1e-3)
    parser.add_argument("--train-hours", type=float, required=False, default=9.0)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--train-sample-limit", type=int, required=False, default=0)
    parser.add_argument("--num-nodes", type=int, required=False, default=1)
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


class PreTraM(pl.LightningModule):
    def __init__(
        self,
        map_encoder,
        agent_encoder,
        n_features,
        projection_dim,
        batch_size,
        lr=1e-4,
        max_epochs=200,
    ) -> None:
        super().__init__()
        self.map_encoder = map_encoder
        self.agent_encoder = agent_encoder

        self.projector = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, projection_dim, bias=False),
        )

        self.loss_fn = NT_XentLoss(batch_size=batch_size, temperature=0.07)
        self.lr = lr
        self.max_epochs = max_epochs

    def forward(self, batch):
        x, _, _ = batch
        map_data, agent_data = x[:, 0:3, ...], x[:, 3:, ...]

        h_i = self.map_encoder(map_data)
        h_j = self.agent_encoder(agent_data)

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
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
    """Based on: https://github.com/Spijkervet/SimCLR"""

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
    
    map_encoder.head = nn.Identity()
    agent_encoder.head = nn.Identity()

    if args.checkpoint:
        map_encoder.load_state_dict(torch.load(args.checkpoint), strict=False)

    pretram = PreTraM(
        map_encoder=map_encoder,
        agent_encoder=agent_encoder,
        n_features=args.encoder_out_dim,
        batch_size=args.batch_size,
        projection_dim=2048,
        lr=args.lr,
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

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        precision=16,
        accelerator="gpu",
        devices=4,
        num_nodes=args.num_nodes,
        max_time={"days": 0, "hours": args.train_hours},
        default_root_dir=args.save_dir,
        callbacks=[lr_monitor],
        logger=csv_logger,
        strategy="ddp",
    )

    trainer.fit(pretram, datamodule=dm)

    save_time = datetime.utcnow().replace(microsecond=0).isoformat()

    torch.save(
        pretram.state_dict(),
        f"{args.save_dir}/models/{args.model}-{save_time}.pt",
    )

    torch.save(
        map_encoder.state_dict(),
        f"{args.save_dir}/models/map-encoder-{args.model}-{save_time}.pt",
    )

    torch.save(
        agent_encoder.state_dict(),
        f"{args.save_dir}/models/agent-encoder-{args.model}-{save_time}.pt",
    )


if __name__ == "__main__":
    main()
  