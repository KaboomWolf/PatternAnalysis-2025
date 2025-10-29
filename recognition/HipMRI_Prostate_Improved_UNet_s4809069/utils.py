import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import Counter

def mean_dice_score(outputs, targets, num_classes):
    """
    Calculate the mean dice score from predicted segmentation and 
    ground truth masks.

    Args:
        outputs (torch.Tensor): logits from model (B, C, H, W)
        targets (torch.Tensor): one hot mask of ground truth
        num_classes (int): number of classes in ground truth

    Returns:
        list[float]: Dice score for each class
    """
    with torch.no_grad():
        # apply soft max and get predicted segmentation
        probs = F.softmax(outputs, dim=1)  # class probabilities
        preds = torch.argmax(probs, dim=1) # predicted class
        targets = torch.argmax(targets, dim=1)
        dices = []

        # compute per class dice scores
        for class_id in range(num_classes):
            pred_mask = (preds == class_id).float()
            target_mask = (targets == class_id).float()
            intersection = (pred_mask * target_mask).sum()
            union = pred_mask.sum() + target_mask.sum()

            # calculate dice score
            if union.item() == 0:
                # Both pred and target empty, give dice of 1
                dice = 1.0
            else:
                dice = (2 * intersection) / union

            dices.append(float(dice))

    return dices

def intersection_union_values(preds, targets, num_classes):
    """
    Compute per-class intersection and union counts from segmentation 
    predictions and one hot ground truth.

    Args:
        preds (torch.tensor): predicted output from model
        targets (torch.Tensor): One hot encoded ground truth
        num_classes (int): Number of classes in ground truth

    Returns:
        intersection (torch.Tensor): tensor with per-class intersection
            counts
        union (torch.Tensor): tensor with per-class union counts
    """
    # logits to predicted class indices - (B,H,W)
    predicted_classes = torch.argmax(F.softmax(preds, dim=1), dim=1)
    # one-hot to class indices - (B,H,W)
    target_classes = torch.argmax(targets, dim=1) 

    # initialise tensors
    intersection = torch.zeros(num_classes, device=preds.device)
    union = torch.zeros(num_classes, device=preds.device)

    # compute per class intersection and union
    for class_id in range(num_classes):
        predicted_mask = (predicted_classes == class_id).float()
        target_mask = (target_classes == class_id).float()

        intersection[class_id] = (predicted_mask * target_mask).sum()
        union[class_id] = predicted_mask.sum() + target_mask.sum()

    return intersection, union

def count_pixels(dataloader, num_classes=6):
    """
    Count the number of pixels for each class in a dataset.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader that yields (image, mask) pairs.
        num_classes (int): Total number of classes in the masks.

    Returns:
        torch.Tensor: Pixel counts per class in masks
    """
    counts = Counter()

    for _, mask in dataloader:
        # Convert one-hot mask to class indices: shape [B, H, W]
        labels = torch.argmax(mask, dim=1)

        # Count pixels per class
        for l in torch.unique(labels):
            counts[int(l)] += (labels == l).sum().item()

    # Convert Counter to tensor
    pixel_counts = torch.zeros(num_classes, dtype=torch.int64)
    for class_id in range(num_classes):
        pixel_counts[class_id] = counts.get(class_id, 0)

    return pixel_counts

def save_checkpoint(checkpoint_dir, epoch, model, optimizer, scheduler):
    """
    Save checkpoints of the model

    Args:
        checkpoint_dir (str): location to save checkpoint
        epoch (int): current epoch of training loop
        model (torch.nn.Module): PyTorch model that needs to be saved
        optimizer (torch.optim.Optimizer): Optimizer used for training
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler used 
            during training
    """
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')

    # check parallel processing
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict() 
    else:
        state_dict = model.state_dict()

    torch.save({
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, ckpt_path)

    print(f"Checkpoint saved at epoch {epoch+1}")


def save_logs(log_dir, epoch, max_epochs, avg_train_loss, mean_train_dices, avg_val_loss, val_dice_per_class, val_mean_dice, lr):
    """
    Prints out logs to the console and saves a copy to the specified file location

    Args:
        log_dir (str): location to save log to
        epoch (int): current training epoch
        max_epochs (int): maximum epochs that training allows
        avg_train_loss (float): average training loss for current epoch
        mean_train_dices (list[float]): per-class mean dice score for training batches
        avg_val_loss (float): average validation loss for current epoch
        val_dice_per_class (list[float]): per-class dice score for validation set
        val_mean_dice (float): mean dice score across classes in validation set
        lr (float): current learning rate
    """
    # print logs in console
    log_str = (
        f"Epoch [{epoch+1}/{max_epochs}] \n"
        f"Train Loss: {avg_train_loss:.4f} \n"
        f"Train Dice: {['{:.3f}'.format(d) for d in mean_train_dices]} \n"
        f"Val Loss: {avg_val_loss:.4f} \n"
        f"Val Dice: {['{:.3f}'.format(d) for d in val_dice_per_class]} \n"
        f"Val Mean Dice: {val_mean_dice:.3f} \n"
        f"LR: {lr:.6f} \n"
    )
    print(log_str, end='')

    # Write logs to file
    log_file = os.path.join(log_dir, f"epoch_{epoch+1}.txt")
    with open(log_file, "w") as f:
        f.write(log_str)

def plot_training_curve(filename, train_losses, val_losses):
    """
    Plot the training and validation curve

    Args:
        filename (str): location to store the graph
        train_losses (list[float]): list of training losses
        val_losses (list[float]): list of validation losses
    """
    plt.figure(figsize=(8,6))
    plt.plot(range(1,len(train_losses)+1), train_losses, label="Train Loss") 
    plt.plot(range(1,len(val_losses)+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch") 
    plt.ylabel("Dice Loss")
    plt.title("Training and Validation Loss") 
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # save image
    plt.savefig(filename, dpi=300, bbox_inches='tight')
