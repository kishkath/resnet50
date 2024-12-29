// Copyright 2018 Oath Inc.
// Licensed under the terms of the MIT license. Please see LICENSE file in project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from model import ResNet50Module
from dataset import ImageNetDataModule


def main(args):
    # Initialize the model
    model = ResNet50Module(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_milestones=args.lr_milestones,
        lr_gamma=args.lr_gamma
    )
    
    # Initialize the data module
    datamodule = ImageNetDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project="resnet50-imagenet",
        log_model=True,
        save_dir=args.output_dir
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="resnet50-{epoch:02d}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=3
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Initialize the trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        strategy=DDPStrategy(find_unused_parameters=False),
        precision="16-mixed",  # Enable mixed precision training
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=50,
        val_check_interval=0.5  # Validate every 0.5 epochs
    )
    
    # Train the model
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet50 on ImageNet")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to ImageNet dataset")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for logs and checkpoints")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size per GPU")
    parser.add_argument("--num_gpus", type=int, default=4,
                        help="Number of GPUs to use")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of data loading workers per GPU")
    parser.add_argument("--max_epochs", type=int, default=90,
                        help="Maximum number of epochs to train")
    
    # Optimization arguments
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--lr_milestones", nargs="+", type=int,
                        default=[500000, 1000000],
                        help="Learning rate milestone steps")
    parser.add_argument("--lr_gamma", type=float, default=0.1,
                        help="Learning rate decay factor")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)