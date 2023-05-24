import torch
import pytorch_lightning as pl

from torch import nn, Tensor
from local_attention import LocalAttention

from .raster_barlow_twins import BarlowTwinsLoss


class REDEncoder(pl.LightningModule):
    """Road Environment Description (RED) Encoder"""

    def __init__(
        self,
        size_encoder_vocab: int = 11,
        num_encoder_layers: int = 6,
        size_decoder_vocab: int = 100,
        num_decoder_layers: int = 6,
        dim_model: int = 512,
        dim_heads_encoder: int = 64,
        dim_attn_window_encoder: int = 64,
        num_heads_decoder: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_encoder_pos: float = 55.0,
        z_dim: int = 512,
        batch_size: int = 8,
        max_train_epochs: int = 200,
        learning_rate=1e-4,
        lambda_coeff=5e-3,
    ):
        super().__init__()
        self.encoder_semantic_embedding = nn.Embedding(
            num_embeddings=size_encoder_vocab,
            embedding_dim=dim_model
            - 2,  # 2 floats for postion in (x, y) (maybe a bit small and should be increased with a Linear layer)
            padding_idx=-1,  # For [pad] token
        )
        self.max_encoder_pos = max_encoder_pos
        self.encoder = LocalTransformerEncoder(
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            dim_heads=dim_heads_encoder,
            dim_attn_window=dim_attn_window_encoder,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.range_decoder_embedding = torch.arange(size_decoder_vocab).expand(
            batch_size, size_decoder_vocab
        )
        self.decoder_semantic_embedding = nn.Embedding(
            num_embeddings=size_decoder_vocab,
            embedding_dim=dim_model - 10,  # For learned pos. embedding
        )
        self.decoder_pos_embedding = nn.Embedding(
            num_embeddings=size_decoder_vocab,
            embedding_dim=10,
        )

        self.decoder = ParallelTransformerDecoder(
            num_layers=num_decoder_layers,
            dim_model=dim_model,
            num_heads=num_heads_decoder,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.projection_head = nn.Sequential(
            nn.Linear(in_features=dim_model, out_features=4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=z_dim),
        )
        self.loss_fn = BarlowTwinsLoss(
            batch_size=batch_size, lambda_coeff=lambda_coeff, z_dim=z_dim
        )
        self.max_epochs = max_train_epochs
        self.learning_rate = learning_rate

    def forward(
        self, idxs_src_tokens: Tensor, pos_src_tokens: Tensor, src_mask: Tensor
    ) -> Tensor:
        pos_src_tokens /= self.max_encoder_pos
        src = torch.concat(
            (self.encoder_semantic_embedding(idxs_src_tokens), pos_src_tokens), dim=2
        )  # Concat in feature dim
        self.range_decoder_embedding = self.range_decoder_embedding.to("cuda")
        tgt = torch.concat(
            (
                self.decoder_semantic_embedding(self.range_decoder_embedding),
                self.decoder_pos_embedding(self.range_decoder_embedding),
            ),
            dim=2,
        )

        return self.decoder(tgt, self.encoder(src, src_mask), src_mask)

    def shared_step(self, batch):
        z_a = self.projection_head(
            self.forward(
                idxs_src_tokens=batch["sample_a"]["idx_src_tokens"],
                pos_src_tokens=batch["sample_a"]["pos_src_tokens"],
                src_mask=batch["src_attn_mask"],
            ).mean(dim=1),
        )
        z_b = self.projection_head(
            self.forward(
                idxs_src_tokens=batch["sample_b"]["idx_src_tokens"],
                pos_src_tokens=batch["sample_b"]["pos_src_tokens"],
                src_mask=batch["src_attn_mask"],
            ).mean(dim=1),
        )
        return self.loss_fn(z_a, z_b)

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


class LocalMultiheadAttention(nn.Module):
    def __init__(
        self, dim_in: int, dim_q: int, dim_k: int, dim_heads: int, dim_attn_window: int
    ):
        super().__init__()
        self.to_q = nn.Linear(dim_in, dim_q)
        self.to_k = nn.Linear(dim_in, dim_k)
        self.to_v = nn.Linear(dim_in, dim_k)
        self.attn = LocalAttention(
            dim=dim_heads,
            window_size=dim_attn_window,
            autopad=True,
            use_rotary_pos_emb=False,
        )

    def forward(self, queries, keys, values, mask):
        q = self.to_q(queries)
        k = self.to_k(keys)
        v = self.to_v(values)

        return self.attn(q, k, v, mask=mask)


class LocalTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        dim_heads: int = 64,
        dim_attn_window: int = 64,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = Residual(
            LocalMultiheadAttention(
                dim_in=dim_model,
                dim_q=dim_model,
                dim_k=dim_model,
                dim_heads=dim_heads,
                dim_attn_window=dim_attn_window,
            ),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor, mask: Tensor) -> Tensor:
        src = self.attention(src, src, src, mask)
        return self.feed_forward(src)


class LocalTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        dim_heads: int = 64,
        dim_attn_window: int = 64,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        add_pos_encoding: bool = False,
    ):
        super().__init__()
        self.add_pos_encoding = add_pos_encoding
        self.layers = nn.ModuleList(
            [
                LocalTransformerEncoderLayer(
                    dim_model, dim_heads, dim_attn_window, dim_feedforward, dropout
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, src: Tensor, mask: Tensor) -> Tensor:
        if self.add_pos_encoding:
            seq_len, dimension = src.size(1), src.size(2)
            src += position_encoding(seq_len, dimension)

        for layer in self.layers:
            src = layer(src, mask)

        return src


class ParallelTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.self_attn = Residual(
            nn.MultiheadAttention(
                embed_dim=dim_model,
                num_heads=num_heads,
                batch_first=True,
            ),
            dimension=dim_model,
            dropout=dropout,
        )
        self.cross_attn = Residual(
            nn.MultiheadAttention(
                embed_dim=dim_model,
                num_heads=num_heads,
                batch_first=True,
            ),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, tgt: Tensor, memory: Tensor, mem_mask: Tensor) -> Tensor:
        tgt, _ = self.self_attn(tgt, tgt, tgt, need_weights=False)
        batch_size, tgt_len = tgt.size(dim=0), tgt.size(dim=1)
        mem_mask = mem_mask[:, None, :].expand(batch_size, tgt_len, -1)
        mem_mask = mem_mask.repeat(1, self.num_heads, 1)
        mem_mask = mem_mask.view(batch_size * self.num_heads, tgt_len, -1)

        tgt, _ = self.cross_attn(
            tgt, memory, memory, attn_mask=mem_mask, need_weights=False
        )

        return self.feed_forward(tgt)


class ParallelTransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        add_pos_encoding: bool = False,
    ):
        super().__init__()
        self.add_pos_encoding = add_pos_encoding

        self.layers = nn.ModuleList(
            [
                ParallelTransformerDecoderLayer(
                    dim_model, num_heads, dim_feedforward, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(dim_model, dim_model)

    def forward(self, tgt: Tensor, memory: Tensor, mem_mask: Tensor) -> Tensor:
        if self.add_pos_encoding:
            seq_len, dimension = tgt.size(1), tgt.size(2)
            tgt += position_encoding(seq_len, dimension)

        for layer in self.layers:
            tgt = layer(tgt, memory, mem_mask)

        return self.linear(tgt)


class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))


def position_encoding(
    seq_len: int,
    dim_model: int,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim / dim_model))

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )
