import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from modules import UNet, CombinedLoss
from dataset import HipMRIDataset
from utils import mean_dice_score, intersection_union_values, count_pixels, save_checkpoint, save_logs, plot_training_curve

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
LR = 1e-4
MAX_EPOCHS = 2
TARGET_DICE = 0.9
NUM_CLASSES = 6
SAVE_PATH = "test_unet_prostate.pth"

# -----------------------------
# Paths
# -----------------------------
# checkpoints
checkpoint_dir = './checkpoints/test_unet_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# logging
log_dir = "./checkpoints/test_logs"
os.makedirs(log_dir, exist_ok=True)

base_img_path = "keras_slices_data/"

# -----------------------------
# Dataset
# -----------------------------
print("> Loading Data...")
train_dataset = HipMRIDataset(f"{base_img_path}keras_slices_train", f"{base_img_path}keras_slices_seg_train")
val_dataset = HipMRIDataset(f"{base_img_path}keras_slices_validate", f"{base_img_path}keras_slices_seg_validate")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -----------------------------
# Calculate Weights
# -----------------------------
print("> Calculating Weights...")
pixel_counts = count_pixels(train_loader, num_classes=6)
weights = 1.0 / pixel_counts
weights = weights / weights.sum() * len(weights)
weights = weights.to(DEVICE)

# -----------------------------
# Model, loss, optimizer
# -----------------------------
print("> Loading Model...")
model = UNet(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
criterion = CombinedLoss(class_weights=weights, dice_weight=1, ce_weight=1)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_EPOCHS, eta_min=1e-7
)

# -----------------------------
# Training loop
# -----------------------------
train_losses = []
val_losses = []

# allow multi-gpu use
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    model = nn.DataParallel(model)

# training loop of all epochs
for epoch in range(MAX_EPOCHS):
    model.train()
    running_loss = 0.0
    train_dices = []
    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"-> Epoch {epoch}, batch {batch_idx} / {len(train_loader)}")
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)

        loss = 0
        for out in outputs:
            loss += criterion(out, masks)
        loss /= len(outputs)

        dices = mean_dice_score(outputs[-1], masks, NUM_CLASSES)
        train_dices.append(dices)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    val_loss = 0.0
    total_intersection = torch.zeros(NUM_CLASSES, device=DEVICE) 
    total_union = torch.zeros(NUM_CLASSES, device=DEVICE)

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)

            loss = 0
            for out in outputs:
                loss += criterion(out, masks)
            loss /= len(outputs)
            val_loss += loss.item()

            # use last output for evaluation
            preds = outputs[-1]
            intersection, union = intersection_union_values(preds, masks, NUM_CLASSES)
            total_intersection += intersection
            total_union += union
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # compute per class dice and mean dice
    dice_per_class = (2 * total_intersection) / total_union
    val_mean_dice = dice_per_class.mean().item()
    val_dice_per_class = dice_per_class.cpu().numpy()
    mean_train_dices = [sum(c)/len(c) for c in zip(*train_dices)]

    # print and save logs
    save_logs(
        log_dir=log_dir,
        epoch=epoch + 1,
        max_epochs=MAX_EPOCHS,
        avg_train_loss=avg_train_loss,
        mean_train_dices=mean_train_dices,
        avg_val_loss=avg_val_loss,
        val_dice_per_class=val_dice_per_class,
        val_mean_dice=val_mean_dice,
        lr=scheduler.get_last_lr()[0]
    )

    # Early stopping
    if all(d >= TARGET_DICE for d in val_dice_per_class):
        print(f"Target Dice {TARGET_DICE} reached for all classes. Stopping training.")
        break

    # Step scheduler
    scheduler.step()

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        save_checkpoint(
            checkpoint_dir=checkpoint_dir, 
            epoch=epoch, 
            model=model,
            optimizer=optimizer, 
            scheduler=scheduler
        )

# -----------------------------
# Save model
# -----------------------------
final_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
torch.save(final_state, SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")
np.savez("loss_history_test.npz", train_losses=train_losses, val_losses=val_losses)

# -----------------------------
# Plot training curve
# -----------------------------
plot_training_curve(
    filename="training_loss_test.png",
    train_losses=train_losses,
    val_losses=val_losses
)