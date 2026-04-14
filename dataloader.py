import os
import torch
import numpy as np
from PIL import Image
import tifffile as tiff
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms  # Crucial import
from sklearn.model_selection import train_test_split

import collections
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable

class OpenEarthMapSarDataset(Dataset):
    def __init__(self, dataset_directory, image_size):
        self.dataset_directory = dataset_directory
        self.optical_path = os.path.join(dataset_directory, "rgb_images")
        self.sar_path = os.path.join(dataset_directory, "sar_images")
        self.label_path = os.path.join(dataset_directory, "labels")

        self.filenames = [f for f in os.listdir(self.optical_path) if f.endswith('.tif')]

        self.image_size = image_size
        
        # Define resize
        self.optical_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        
        self.label_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST)  # Keep label integers
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
        
        # convert to single label classifcation
        label = self.label_transform(label_images)
        label_np = np.array(label)
        mode_label = np.argmax(np.bincount(label_np.flatten()))
        label = torch.tensor(mode_label -1).long()
        # 

        # convert to tensor
        optical = transforms.ToTensor()(optical_images)
        optical = self.optical_transform(optical_images)

        sar = torch.from_numpy(sar_images).float()
        sar = sar / 255.0
        sar = torch.nn.functional.interpolate(sar.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0) #resize
        # sar = transforms.ToTensor()(sar)
    
        # segmentation
        # label = torch.from_numpy(np.array(label_images)).long()
        # label = self.label_transform(label_images)
        # label = torch.from_numpy(np.array(label)).long()

        # transforms

        return optical, sar, label

def get_dataloader(config):
    data = OpenEarthMapSarDataset(config.dataset_root, config.image_size)
    indices = np.arange(len(data))

    train_index, temp_index = train_test_split(indices, train_size=config.train_size, random_state=42, shuffle=True)
    val_index, test_index = train_test_split(temp_index, train_size=config.val_size, random_state=42, shuffle=True)

    train_loader = DataLoader(Subset(data, train_index), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(data, val_index), batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(data, test_index), batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # from config import Config
    # config = Config()
    # dataset = OpenEarthMapSarDataset(config.dataset_root)
    
    # # Check the first 5 images
    # for i in range(5):
    #     opt, sar, lbl = dataset[i]
    #     print(f"Sample {i}: Optical shape {opt.shape}, SAR shape {sar.shape}")

    import tifffile
    sample_file = r".\dataset\openearthmap-sar\train\sar_images\TrainArea_003.tif"

    # If you don't know a filename, let's grab the first one automatically:
    # sample_dir = r".\dataset\openearthmap-sar\train\sar_images"
    # sample_file = os.path.join(sample_dir, os.listdir(sample_dir)[0])

    with tifffile.TiffFile(sample_file) as tif:
        data = tif.asarray()
        print(f"File: {os.path.basename(sample_file)}")
        print(f"Number of series/pages: {len(tif.series)}")
        print(f"Shape: {data.shape}")

        # ONE channel? dataset should have been 2