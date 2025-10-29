import torch
import torch.nn.functional as F
import os 

from torch.utils.data import DataLoader
from modules import UNet
from dataset import HipMRIDataset
from utils import intersection_union_values, colorize_mask, save_comparison_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
BATCH_SIZE = 8
MODEL_PATH = "./unet_prostate2.pth"
SAVE_DIR = "./test_predictions"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------
# Load dataset
# ---------------------
print("> Loading dataset...")
base_path = "keras_slices_data/"
test_dataset = HipMRIDataset(f"{base_path}keras_slices_test", f"{base_path}keras_slices_seg_test")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# -------------------
# Load Model
# -------------------
print(f"> Loading Model from {MODEL_PATH}")
model = UNet(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict=state_dict)
model.eval()

# -------------------
# Eval loop
# -------------------
total_intersection = torch.zeros(NUM_CLASSES, device=DEVICE)
total_union = torch.zeros(NUM_CLASSES, device=DEVICE)

with torch.no_grad():
    sample_idx = 0  
    for batch_idx, (images, masks) in enumerate(test_loader):
        print(f"-> Batch {batch_idx} / {len(test_loader)}")
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        outputs = (model(images))[-1]

        intersection, union = intersection_union_values(outputs, masks, NUM_CLASSES)
        total_intersection += intersection
        total_union += union

        # Visualization
        probs = F.softmax(outputs, dim=1)
        preds_idx = torch.argmax(probs, dim=1).cpu().numpy()
        targets_idx = torch.argmax(masks, dim=1).cpu().numpy() 
        images_np = images.cpu().numpy()                       

        for idx in range(images_np.shape[0]):
            save_comparison_image(
                filename=os.path.join(SAVE_DIR, f"sample_{sample_idx:04d}.png"),
                img_gray=images_np[idx, 0],
                ground_truth_rgb=colorize_mask(targets_idx[idx]),
                pred_rgb=colorize_mask(preds_idx[idx])
            )
            sample_idx += 1 

# ------------------
# Dice Scores
# ------------------
dice_per_class = (2 * total_intersection) / (total_union)
mean_dice = dice_per_class.mean().item()

log_str = (
    f"Dice Per Class: {['{:.3f}'.format(d) for d in dice_per_class.cpu().numpy()]}\n"
    f"Mean Dice: {mean_dice:.3f}"
)

print(log_str)