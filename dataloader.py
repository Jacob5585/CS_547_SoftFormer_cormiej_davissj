import os
import torch
import numpy as np
from PIL import Image
import tifffile as tiff
from torch.utils.data import Dataset, DataLoader, Subset, default_collate
from torchvision import transforms, tv_tensors
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
import patching
from functools import partial

import collections
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable

# Moved do to windows
def custom_collate(batch, config):
        opt, sar, label = default_collate(batch)

        if config.method == "classification":
            opt = patching.pre_patch_batch(opt, config.patch_size)
            sar = patching.pre_patch_batch(sar, config.patch_size)
            label = patching.patchify_labels(label, config.patch_size)
        
        return opt, sar, label

class AugmentatSubset(Subset):
    def __init__(self, dataset, indices, augment=False):
        super().__init__(dataset, indices)
        self.augment = augment

    def __getitem__(self, idx):
        # Temporarily enable augmentation on the base dataset
        self.dataset.augment = self.augment
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]

class OpenEarthMapSarDataset(Dataset):
    def __init__(self, dataset_directory, image_size):
        self.dataset_directory = dataset_directory
        self.optical_path = os.path.join(dataset_directory, "rgb_images")
        self.sar_path = os.path.join(dataset_directory, "sar_images")
        self.label_path = os.path.join(dataset_directory, "labels")

        self.filenames = [f for f in os.listdir(self.optical_path) if f.endswith('.tif')]
        self.image_size = image_size
        self.augment = False
        
        # Define resize
        self.optical_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        
        self.label_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST)  # Keep label integers
        ])

        self.base_agumentation = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
        ])

        self.optical_augumentation = v2.Compose([
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.01),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        fname = self.filenames[index]

        # load data
        optical_images = Image.open(os.path.join(self.optical_path, fname)).convert('RGB') # 0-255 3 channels
        
        sar_images = tiff.imread(os.path.join(self.sar_path, fname)) #  0-255 2 channels
        # if sar_images.shape[-1] == 2:
        #     sar_images = sar_images.transpose(2, 0, 1) # Channels, Height, Width
        if sar_images.ndim == 2:
            # Change (H, W) -> (1, H, W)
            sar_images = sar_images[np.newaxis, :, :]
        elif sar_images.shape[-1] == 2 or sar_images.shape[-1] == 3:
            # Change (H, W, C) -> (C, H, W)
            sar_images = sar_images.transpose(2, 0, 1)
        
        label_images = Image.open(os.path.join(self.label_path, fname))

        # convert to tensor
        optical = transforms.ToTensor()(optical_images)
        optical = self.optical_transform(optical_images)

        sar = torch.from_numpy(sar_images).float()
        sar = sar / 255.0
        sar = torch.nn.functional.interpolate(sar.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0) #resize
        # sar = transforms.ToTensor()(sar)
    
        label = self.label_transform(label_images)
        label = torch.from_numpy(np.array(label)).long() - 1

        # agumentations
        if self.augment:
            optical = tv_tensors.Image(optical)
            sar = tv_tensors.Image(sar)
            label = tv_tensors.Mask(label)

            optical = self.optical_augumentation(optical)

            if torch.rand(1) > 0.5:
                noise = 1.0 + torch.randn_like(sar) * 0.05
                sar = torch.clamp(sar * noise, 0, 1)

            optical, sar, label = self.base_agumentation(optical, sar, label)

        # return optical, sar, label
        return optical.as_subclass(torch.Tensor), sar.as_subclass(torch.Tensor), label.squeeze(0).as_subclass(torch.Tensor)

def get_dataloader(config):
    data = OpenEarthMapSarDataset(config.dataset_root, config.image_size)
    indices = np.arange(len(data))

    train_index, temp_index = train_test_split(indices, train_size=config.train_size, random_state=42, shuffle=True)
    val_index, test_index = train_test_split(temp_index, train_size=config.val_size, random_state=42, shuffle=True)

    collate_fn = partial(custom_collate, config=config)

    train_loader = DataLoader(
        AugmentatSubset(data, train_index, augment=True),
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        AugmentatSubset(data, val_index, augment=False), 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )

    test_loader = DataLoader(
        AugmentatSubset(data, test_index, augment=False), 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )

    return train_loader, val_loader, test_loader


# if __name__ == "__main__":
#     # from config import Config
#     # config = Config()
#     # dataset = OpenEarthMapSarDataset(config.dataset_root)
    
#     # # Check the first 5 images
#     # for i in range(5):
#     #     opt, sar, lbl = dataset[i]
#     #     print(f"Sample {i}: Optical shape {opt.shape}, SAR shape {sar.shape}")

#     import tifffile
#     sample_file = r".\dataset\openearthmap-sar\train\sar_images\TrainArea_003.tif"

#     # sample_dir = r".\dataset\openearthmap-sar\train\sar_images"
#     # sample_file = os.path.join(sample_dir, os.listdir(sample_dir)[0])

#     with tifffile.TiffFile(sample_file) as tif:
#         data = tif.asarray()
#         print(f"File: {os.path.basename(sample_file)}")
#         print(f"Number of series/pages: {len(tif.series)}")
#         print(f"Shape: {data.shape}")

#         # ONE channel? dataset should have been 2