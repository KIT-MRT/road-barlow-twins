import torch
import pytorch_lightning as pl

from torch import nn, Tensor
from local_attention import LocalAttention

from .raster_barlow_twins import BarlowTwinsLoss
from .dual_motion_vit import pytorch_neg_multi_log_likelihood_batch


class REDEncoder(pl.LightningModule):
    """Road Environment Description (RED) Encoder"""

    def __init__(
        self,
        size_encoder_vocab: int = 11,
        dim_encoder_semantic_embedding: int = 4,
        num_encoder_layers: int = 6,
        size_decoder_vocab: int = 100,
        num_decoder_layers: int = 6,
        dim_model: int = 512,
        dim_heads_encoder: int = 64,
        dim_attn_window_encoder: int = 64,
        num_heads_decoder: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_dist: float = 50.0,
        z_dim: int = 512,
        batch_size: int = 8,
        max_train_epochs: int = 200,
        learning_rate=1e-4,
        lambda_coeff=5e-3,
    ):
        super().__init__()
        self.encoder_semantic_embedding = nn.Embedding(
            num_embeddings=size_encoder_vocab,
            embedding_dim=dim_encoder_semantic_embedding,
            padding_idx=-1,  # For [pad] token
        )
        self.to_dim_model = nn.Linear(
            in_features=dim_encoder_semantic_embedding + 2,  # For position as (x, y)
            out_features=dim_model,
        )
        self.max_dist = max_dist
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
            nn.Linear(
                in_features=size_decoder_vocab * 2, out_features=4096
            ),  # Mean, var per token
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
        pos_src_tokens /= self.max_dist
        src = torch.concat(
            (self.encoder_semantic_embedding(idxs_src_tokens), pos_src_tokens), dim=2
        )  # Concat in feature dim
        src = self.to_dim_model(src)

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
        road_env_tokens_a = self.forward(
            idxs_src_tokens=batch["sample_a"]["idx_src_tokens"],
            pos_src_tokens=batch["sample_a"]["pos_src_tokens"],
            src_mask=batch["src_attn_mask"],
        )
        road_env_tokens_b = self.forward(
            idxs_src_tokens=batch["sample_b"]["idx_src_tokens"],
            pos_src_tokens=batch["sample_b"]["pos_src_tokens"],
            src_mask=batch["src_attn_mask"],
        )
        z_a = self.projection_head(
            torch.concat(
                (road_env_tokens_a.mean(dim=2), road_env_tokens_a.var(dim=2)), dim=1
            )
        )
        z_b = self.projection_head(
            torch.concat(
                (road_env_tokens_b.mean(dim=2), road_env_tokens_b.var(dim=2)), dim=1
            )
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


class EgoTrajectoryEncoder(nn.Module):
    def __init__(
        self,
        dim_semantic_embedding=4,
        max_dist=50.0,
        num_timesteps=11,
        num_layers=6,
        dim_model=128,
        num_heads=8,
        dim_feedforward=512,
        dropout=0.1,
        dim_output=256,
    ):
        super().__init__()
        self.to_dim_model = nn.Linear(
            in_features=dim_semantic_embedding + 3,  # 2 pos, 1 temp
            out_features=dim_model,
        )
        self.semantic_embedding = nn.Embedding(
            num_embeddings=6,  # Classes static + dynamic
            embedding_dim=dim_semantic_embedding,
        )
        self.max_dist = max_dist
        self.num_timesteps = num_timesteps

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(dim_model, dim_output)

    def forward(self, idxs_semantic_embedding, pos_src_tokens):
        pos_src_tokens /= self.max_dist
        batch_size = idxs_semantic_embedding.size(dim=0)
        time_encoding = torch.arange(0, self.num_timesteps) / (self.num_timesteps - 1)
        time_encoding = time_encoding.expand(batch_size, -1)[:, :, None]
        time_encoding = time_encoding.to("cuda")

        src = torch.concat(
            (
                self.semantic_embedding(idxs_semantic_embedding),
                pos_src_tokens,
                time_encoding,
            ),
            dim=2,
        )
        src = self.to_dim_model(src)

        for layer in self.layers:
            src = layer(src, src, src)

        return self.linear(src)


class REDMotionPredictor(pl.LightningModule):
    def __init__(
        self,
        dim_road_env_encoder,
        dim_road_env_attn_window,
        dim_ego_trajectory_encoder,
        num_trajectory_proposals,
        prediction_horizon,
        batch_size,
        learning_rate,
        auxiliary_rbt_loss=False,
        auxiliary_loss_weight=0.3,
        prediction_subsampling_rate=1,
        num_fusion_layers=6,
    ):
        super().__init__()
        self.num_trajectory_proposals = num_trajectory_proposals
        self.prediction_horizon = prediction_horizon
        self.prediction_subsampling_rate = prediction_subsampling_rate
        self.lr = learning_rate

        self.road_env_encoder = REDEncoder(
            dim_model=dim_road_env_encoder,
            dim_attn_window_encoder=dim_road_env_attn_window,
            batch_size=batch_size,
        )
        self.ego_trajectory_encoder = EgoTrajectoryEncoder(
            dim_model=dim_ego_trajectory_encoder,
            dim_output=dim_road_env_encoder,
        )
        self.fusion_block = REDFusionBlock(
            dim_model=dim_road_env_encoder, num_layers=num_fusion_layers
        )  # Opt. test w/ more
        self.motion_head = nn.Sequential(
            nn.LayerNorm((dim_road_env_encoder,), eps=1e-06, elementwise_affine=True),
            nn.Linear(
                in_features=dim_road_env_encoder,
                out_features=num_trajectory_proposals
                * 2
                * (prediction_horizon // prediction_subsampling_rate)
                + num_trajectory_proposals,
            ),  # Multiple trajectory proposals with (x, y) every (0.1 sec // subsampling rate) and confidences
        )
        self.auxiliary_rbt_loss = auxiliary_rbt_loss
        self.auxiliary_loss_weight = auxiliary_loss_weight

    def forward(
        self,
        env_idxs_src_tokens: Tensor,
        env_pos_src_tokens: Tensor,
        env_src_mask: Tensor,
        ego_idxs_semantic_embedding: Tensor,
        ego_pos_src_tokens: Tensor,
    ):
        road_env_tokens = self.road_env_encoder(
            env_idxs_src_tokens, env_pos_src_tokens, env_src_mask
        )
        ego_trajectory_tokens = self.ego_trajectory_encoder(
            ego_idxs_semantic_embedding, ego_pos_src_tokens
        )
        fused_tokens = self.fusion_block(
            q=ego_trajectory_tokens,
            k=road_env_tokens,
            v=road_env_tokens,
        )
        motion_embedding = self.motion_head(
            fused_tokens.mean(dim=1)
        )  # Sim. to improved ViT global avg pooling before classification
        confidences_logits, logits = (
            motion_embedding[:, : self.num_trajectory_proposals],
            motion_embedding[:, self.num_trajectory_proposals :],
        )
        logits = logits.view(
            -1,
            self.num_trajectory_proposals,
            (self.prediction_horizon // self.prediction_subsampling_rate),
            2,
        )

        return confidences_logits, logits

    def _shared_step(self, batch, batch_idx):
        # x, y, is_available = batch
        is_available = batch["future_ego_trajectory"]["is_available"]
        y = batch["future_ego_trajectory"]["trajectory"]

        env_idxs_src_tokens = batch["sample_a"]["idx_src_tokens"]
        env_pos_src_tokens = batch["sample_a"]["pos_src_tokens"]
        env_src_mask = batch["src_attn_mask"]
        ego_idxs_semantic_embedding = batch["past_ego_trajectory"][
            "idx_semantic_embedding"
        ]
        ego_pos_src_tokens = batch["past_ego_trajectory"]["pos_src_tokens"]

        y = y[
            :,
            (
                self.prediction_subsampling_rate - 1
            ) : self.prediction_horizon : self.prediction_subsampling_rate,
            :,
        ]
        is_available = is_available[
            :,
            (
                self.prediction_subsampling_rate - 1
            ) : self.prediction_horizon : self.prediction_subsampling_rate,
        ]

        if self.auxiliary_rbt_loss:
            road_env_tokens_a = self.road_env_encoder(
                idxs_src_tokens=batch["sample_a"]["idx_src_tokens"],
                pos_src_tokens=batch["sample_a"]["pos_src_tokens"],
                src_mask=batch["src_attn_mask"],
            )
            road_env_tokens_b = self.road_env_encoder(
                idxs_src_tokens=batch["sample_b"]["idx_src_tokens"],
                pos_src_tokens=batch["sample_b"]["pos_src_tokens"],
                src_mask=batch["src_attn_mask"],
            )
            env_z_a = self.road_env_encoder.projection_head(
                torch.concat(
                    (road_env_tokens_a.mean(dim=2), road_env_tokens_a.var(dim=2)), dim=1
                )
            )
            env_z_b = self.road_env_encoder.projection_head(
                torch.concat(
                    (road_env_tokens_b.mean(dim=2), road_env_tokens_b.var(dim=2)), dim=1
                )
            )
            rbt_loss = self.road_env_encoder.loss_fn(env_z_a, env_z_b)

            ego_trajectory_tokens = self.ego_trajectory_encoder(
                ego_idxs_semantic_embedding, ego_pos_src_tokens
            )
            fused_tokens = self.fusion_block(
                q=ego_trajectory_tokens,
                k=road_env_tokens_a,
                v=road_env_tokens_a,
            )
            motion_embedding = self.motion_head(fused_tokens.mean(dim=1))
            confidences_logits, logits = (
                motion_embedding[:, : self.num_trajectory_proposals],
                motion_embedding[:, self.num_trajectory_proposals :],
            )
            logits = logits.view(
                -1,
                self.num_trajectory_proposals,
                (self.prediction_horizon // self.prediction_subsampling_rate),
                2,
            )
            motion_loss = pytorch_neg_multi_log_likelihood_batch(
                y, logits, confidences_logits, is_available
            )

            loss = motion_loss + self.auxiliary_loss_weight * rbt_loss
        else:
            confidences_logits, logits = self.forward(
                env_idxs_src_tokens,
                env_pos_src_tokens,
                env_src_mask,
                ego_idxs_semantic_embedding,
                ego_pos_src_tokens,
            )

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


class REDFusionBlock(nn.Module):
    def __init__(
        self,
        num_layers=3,
        num_heads=8,
        dim_model=128,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, q, k, v):
        for layer in self.layers:
            q = layer(q, k, v)

        return q


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_model, num_heads, dim_feedforward, dropout):
        super().__init__()
        self.attn = Residual(
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

    def forward(self, src_q, src_k, src_v):
        attn_out = self.attn(src_q, src_k, src_v, need_weights=False)
        return self.feed_forward(attn_out)


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
        tgt = self.self_attn(tgt, tgt, tgt, need_weights=False)
        batch_size, tgt_len = tgt.size(dim=0), tgt.size(dim=1)
        mem_mask = mem_mask[:, None, :].expand(batch_size, tgt_len, -1)
        mem_mask = mem_mask.repeat(1, self.num_heads, 1)
        mem_mask = mem_mask.view(batch_size * self.num_heads, tgt_len, -1)

        tgt = self.cross_attn(
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

    def forward(self, *tensors: Tensor, **kwargs: dict) -> Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        output_sublayer = self.sublayer(*tensors, **kwargs)

        # nn.MultiheadAttention always returns a tuple (out, attn_weights or None)
        if isinstance(output_sublayer, tuple):
            output_sublayer = output_sublayer[0]

        return self.norm(tensors[0] + self.dropout(output_sublayer))


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


def red_motion_inference(model, batch, device="cuda", return_all_numpy=False):
    is_available = batch["future_ego_trajectory"]["is_available"].to(device)
    y = batch["future_ego_trajectory"]["trajectory"].to(device)

    env_idxs_src_tokens = batch["sample_a"]["idx_src_tokens"].to(device)
    env_pos_src_tokens = batch["sample_a"]["pos_src_tokens"].to(device)
    env_src_mask = batch["src_attn_mask"].to(device)
    ego_idxs_semantic_embedding = batch["past_ego_trajectory"][
        "idx_semantic_embedding"
    ].to(device)
    ego_pos_src_tokens = batch["past_ego_trajectory"]["pos_src_tokens"].to(device)

    confidences_logits, logits = model(
        env_idxs_src_tokens,
        env_pos_src_tokens,
        env_src_mask,
        ego_idxs_semantic_embedding,
        ego_pos_src_tokens,
    )
    confidences = torch.softmax(confidences_logits, dim=1)

    if not return_all_numpy:
        return confidences, logits
    else:
        logits_np = logits.squeeze(0).cpu().numpy()
        confidences_np = confidences.squeeze(0).cpu().numpy()
        is_available_np = is_available.squeeze(0).long().cpu().numpy()
        y_np = y.squeeze(0).cpu().numpy()

        return logits_np, confidences_np, is_available_np, y_np
