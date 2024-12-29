import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import TQDMProgressBar
from model import ResNet50Module
from dataset import ImageNetDataModule

# Kaggle-specific paths
KAGGLE_INPUT_DIR = "/kaggle/input/imagenetmini-1000/imagenet-mini"  # Adjust if your dataset path is different
KAGGLE_OUTPUT_DIR = "/kaggle/working/resnet50-output"

# Create output directory
os.makedirs(KAGGLE_OUTPUT_DIR, exist_ok=True)

# Initialize the model
model = ResNet50Module(
    learning_rate=0.01,
    weight_decay=1e-4,
    lr_milestones=[500000, 1000000],
    lr_gamma=0.1
)

# Initialize the data module
datamodule = ImageNetDataModule(
    data_dir=KAGGLE_INPUT_DIR,
    batch_size=64,  # Reduced batch size for Kaggle GPU
    num_workers=2   # Reduced workers for Kaggle
)

# Callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(KAGGLE_OUTPUT_DIR, "checkpoints"),
    filename="resnet50-{epoch:02d}-{val_acc:.3f}",
    monitor="val_acc",
    mode="max",
    save_top_k=3
)

lr_monitor = LearningRateMonitor(logging_interval="step")
progress_bar = TQDMProgressBar(refresh_rate=20)  # Show progress bar with updates every 20 batches

# Initialize the trainer
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,  # Kaggle typically provides 1 GPU
    precision="16-mixed",  # Enable mixed precision training
    max_epochs=30,  # Reduced epochs for Kaggle time limits
    callbacks=[checkpoint_callback, lr_monitor, progress_bar],
    enable_progress_bar=True,
    log_every_n_steps=50,
    val_check_interval=0.5
)

# Train the model
trainer.fit(model, datamodule=datamodule) 
