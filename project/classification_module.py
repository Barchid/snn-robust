from argparse import ArgumentParser
from os import times

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from spikingjelly.clock_driven import functional
import torchmetrics

from project.models.spiking_lenet5 import SpikingLeNet5


class ClassificationModule(pl.LightningModule):
    def __init__(self, in_channels: int, num_classes: int, learning_rate: float, neuron_model: str, bias: bool, timesteps: int, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["timesteps", "in_channels", "num_classes"])

        self.learning_rate = self.hparams.learning_rate

        self.model = SpikingLeNet5(
            in_channels=in_channels,
            num_classes=num_classes,
            neuron_model=self.hparams.neuron_model,
            bias=self.hparams.bias
        )

    def forward(self, x):
        # # Reshapes the input tensor from (B, T, C, H, W) to (T, B, C, H, W).
        # x = x.permute(1, 0, 2, 3, 4)

        # IMPORTANT: always apply reset_net before a new forward
        functional.reset_net(self.model)

        # (T, B, C, H, W) --> (B, num_classes)
        x = self.model(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torchmetrics.functional.accuracy(y_hat.clone().detach(), y)

        self.log('train_loss', loss, on_epoch=True, prog_bar=False)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torchmetrics.functional.accuracy(y_hat.clone().detach(), y)

        # logs
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torchmetrics.functional.accuracy(y_hat.clone().detach(), y)

        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # Here, you add every arguments needed for your module
        # NOTE: they must appear as arguments in the __init___() function
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--bias', action="store_true")
        parser.add_argument('--neuron_model', choices=["LIF", "PLIF", "IF"], default="LIF")
        return parser
