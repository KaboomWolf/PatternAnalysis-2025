import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch.nn.functional as F

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    affines = []
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # HipMRI quirk

    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames, disable=True)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage
        affines.append(affine)
        if i > 20 and early_stop:
            break
    return (images, affines) if getAffines else images

def to_onehot(arr, num_classes=6, dtype=np.float32):
    """
    Convert class index to one hot encoded mask.

    Args:
        arr (np.ndarray): Input mask with integer class labels
        num_classes (int, optional): Number of classes in ground truth.
        dtype (_type_, optional): Data type of the output one-hot array.
             Defaults to np.float32.

    Returns:
        np.ndarray: One-hot encoded array
    """
    shape = arr.shape

    # initialise empty array to hold one-hot mask
    masks = np.zeros((num_classes, *shape), dtype=dtype)

    # add each one hot mask to array
    for class_id in range(num_classes):
        masks[class_id][arr == class_id] = 1
    
    return masks

def resize_image(img, size=(256, 128), mode='bilinear'):
    """
    Resizes image to the specified size with PyTorch interpolation.

    Args:
        img (torch.Tensor or np.ndarray): Input image
        size (tuple, optional): target size of image. Defaults to (256, 128).
        mode (str, optional): Interpolation mode. Defaults to 'bilinear'.

    Returns:
        torch.Tensor: Resized image tensor
    """
    # convert np array to tensor
    if isinstance(img, np.ndarray):
        img = torch.tensor(img, dtype=torch.float32)
    
    # turn [C, H, W] into [N, C, H, W]
    img = img.unsqueeze(0)

    # align corners for modes that allow it
    if mode in ['bilinear', 'linear', 'bicubic', 'trilinear']:
        img = F.interpolate(img, size=size, mode=mode, align_corners=True)
    else:
        img = F.interpolate(img, size=size, mode=mode)

    # remove extra dimension needed for interpolate
    img = img.squeeze(0)
    return img


class HipMRIDataset(Dataset):
    """
    HipMRI dataset for Hip MRI and segmentation masks.

    """
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        # load image files
        self.img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.nii.gz')])
        self.mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # load image
        img_path = self.img_files[idx]
        img = load_data_2D([img_path], normImage=True)[0]

        # add channel dimension
        img = np.expand_dims(img, axis=0) # add channel for [C, H, W]
        img = resize_image(img, size=(256, 128), mode='bilinear')
        img_tensor = torch.tensor(img, dtype=torch.float32)

        # convert between img and mask names
        img_name = os.path.basename(img_path)
        mask_name = img_name.replace("case_", "seg_")
        mask_path = os.path.join(self.mask_dir, mask_name)

        # load mask
        mask = load_data_2D([mask_path], categorical=False, dtype=np.float32)[0]
        mask = to_onehot(mask, num_classes=6)

        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=(256,128), mode='nearest').squeeze(0)
        mask_tensor = (mask_tensor > 0.5).float()

        # apply transforms
        if self.transform:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        return img_tensor, mask_tensor