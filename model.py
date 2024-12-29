import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.lr_scheduler import MultiStepLR
from typing import Optional, Dict, Any
import time


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
        
        print("\n[INIT] Initializing ResNet50 Model...")
        print(f"[INIT] Learning Rate: {learning_rate}")
        print(f"[INIT] Weight Decay: {weight_decay}")
        print(f"[INIT] LR Milestones: {lr_milestones}")
        print(f"[INIT] LR Gamma: {lr_gamma}")
        
        # Initialize model with pretrained weights
        print("[INIT] Loading pretrained ResNet50 weights...")
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        print("[INIT] ResNet50 model loaded successfully!")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize validation metrics
        self.validation_step_outputs = []
        
        # Training metrics
        self.train_start_time = None
        self.epoch_start_time = None
        self.batch_start_time = None
        
    def on_train_start(self):
        print("\n[TRAINING] Starting training process...")
        print(f"[TRAINING] Device being used: {self.device}")
        self.train_start_time = time.time()
        
    def on_train_epoch_start(self):
        print(f"\n[EPOCH {self.current_epoch}] Starting epoch...")
        self.epoch_start_time = time.time()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.batch_start_time = time.time()
            
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy and time
        acc = (logits.argmax(dim=1) == y).float().mean()
        current_lr = self.optimizers().param_groups[0]['lr']
        
        # Log metrics to progress bar
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        self.log('learning_rate', current_lr, prog_bar=True)
        
        # Print detailed metrics every 50 batches
        if batch_idx % 50 == 0:
            batch_time = time.time() - self.batch_start_time
            print(f"\n[EPOCH {self.current_epoch}][BATCH {batch_idx}] Stats:")
            print(f"  - Loss: {loss:.4f}")
            print(f"  - Accuracy: {acc:.4f}")
            print(f"  - Learning Rate: {current_lr:.6f}")
            print(f"  - Batch Time: {batch_time:.2f}s")
            print(f"  - Samples Processed: {(batch_idx + 1) * self.trainer.datamodule.batch_size}")
            self.batch_start_time = time.time()
        
        return loss
    
    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        total_time = time.time() - self.train_start_time
        print(f"\n[EPOCH {self.current_epoch}] Epoch completed!")
        print(f"  - Epoch Time: {epoch_time:.2f}s")
        print(f"  - Total Training Time: {total_time/3600:.2f}h")
    
    def on_validation_start(self):
        print(f"\n[VALIDATION] Starting validation for epoch {self.current_epoch}...")
        self.validation_start_time = time.time()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        # Log metrics to progress bar
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        # Print progress every 20 validation batches
        if batch_idx % 20 == 0:
            print(f"[VALIDATION] Batch {batch_idx} - Loss: {loss:.4f}, Acc: {acc:.4f}")
        
        # Store outputs for epoch end
        self.validation_step_outputs.append({'val_loss': loss, 'val_acc': acc})
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def on_validation_epoch_end(self):
        # Calculate average validation metrics
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in self.validation_step_outputs]).mean()
        
        # Calculate validation time
        validation_time = time.time() - self.validation_start_time
        
        # Print detailed validation results
        print(f"\n[VALIDATION] Epoch {self.current_epoch} Results:")
        print(f"  - Average Loss: {avg_loss:.4f}")
        print(f"  - Average Accuracy: {avg_acc:.4f}")
        print(f"  - Validation Time: {validation_time:.2f}s")
        print("-" * 50)
        
        # Clear the validation step outputs
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        print("\n[OPTIMIZER] Configuring optimizer and scheduler...")
        
        # Optimizer with weight decay
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay
        )
        print(f"[OPTIMIZER] SGD configured with:")
        print(f"  - Learning rate: {self.hparams.learning_rate}")
        print(f"  - Momentum: 0.9")
        print(f"  - Weight decay: {self.hparams.weight_decay}")
        
        # Learning rate scheduler
        scheduler = MultiStepLR(
            optimizer,
            milestones=self.hparams.lr_milestones,
            gamma=self.hparams.lr_gamma
        )
        print(f"[SCHEDULER] MultiStepLR configured with:")
        print(f"  - Milestones: {self.hparams.lr_milestones}")
        print(f"  - Gamma: {self.hparams.lr_gamma}")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }