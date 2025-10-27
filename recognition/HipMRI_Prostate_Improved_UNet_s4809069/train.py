from torch.utils.data import DataLoader
from dataset import HipMRIDataset

if __name__ == "__main__":
    train_dataset = HipMRIDataset(
        img_dir='keras_slices_data/keras_slices_train',
        mask_dir='keras_slices_data/keras_slices_seg_train',
    )
    train_loader = DataLoader(train_dataset, batch_size=4)
   
    validate_dataset = HipMRIDataset(
        img_dir='keras_slices_data/keras_slices_validate',
        mask_dir='keras_slices_data/keras_slices_seg_validate',
    )
    validate_loader = DataLoader(validate_dataset, batch_size=4)

    test_dataset = HipMRIDataset(
        img_dir='keras_slices_data/keras_slices_test',
        mask_dir='keras_slices_data/keras_slices_seg_test',
    )
    test_loader = DataLoader(test_dataset, batch_size=4)
