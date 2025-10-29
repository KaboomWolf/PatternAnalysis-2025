import torch
import torch.nn.functional as F

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