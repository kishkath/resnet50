import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.lr_scheduler import MultiStepLR
from typing import Optional, Dict, Any


class ResNet50Module(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        lr_milestones: list = [500000, 1000000],
        lr_gamma: float = 0.1
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model with pretrained weights
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Log metrics
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Log metrics
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # Optimizer with weight decay
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = MultiStepLR(
            optimizer,
            milestones=self.hparams.lr_milestones,
            gamma=self.hparams.lr_gamma
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }