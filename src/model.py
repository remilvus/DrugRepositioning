import torch
from torch import optim
import torch.nn as nn
import pytorch_lightning as pl


class SmilesTransformer(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        embed_size=32,
        num_layers=4,
        num_heads=8,
        hidden_size=1024,
        dropout=0.1,
    ):
        super(SmilesTransformer, self).__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                dropout=dropout,
            ),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(embed_size, 1)

    def forward(self, x):
        x_embed = self.embedding(x) * (self.embedding.embedding_dim**0.5)
        transformer_output = self.transformer(x_embed)
        logits = self.fc(transformer_output.mean(dim=1))
        return logits.squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        assert not torch.isnan(y_pred).any()
        loss = (y_pred - y) ** 2
        loss = loss.mean()

        # Logging to TensorBoard (if installed) by default
        self.log("train loss", loss)
        self.log("train RMSE", loss**0.5)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        assert not torch.isnan(y_pred).any()
        loss = (y_pred - y) ** 2
        loss = loss.mean()

        # Logging to TensorBoard (if installed) by default
        self.log("val loss", loss)
        self.log("val RMSE", loss**0.5)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
