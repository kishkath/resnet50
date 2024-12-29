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
        
        # Calculate accuracy
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        # Log metrics to progress bar
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        # Print metrics every 100 batches
        if batch_idx % 100 == 0:
            self.print(f'Epoch {self.current_epoch} | Batch {batch_idx} | Loss: {loss:.4f} | Acc: {acc:.4f}')
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        # Log metrics to progress bar
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        # Calculate average validation metrics
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        
        # Print validation results
        self.print(f'\nValidation Epoch {self.current_epoch} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}\n')
    
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