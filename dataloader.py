import os
# import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

class OpenEarthMapSarDataset(Dataset):
    def __init__(self, dataset_directory):
        self.dataset_directory = dataset_directory
        self.optical_path = os.path.join(dataset_directory, "rgb_images")
        self.sar_path = os.path.join(dataset_directory, "sar_images")
        self.label_path = os.path.join(dataset_directory, "label_images")

        self.filenames = [f for f in os.listdir(self.opt_dir) if f.endswith('.tif')]


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        fname = self.filenames[index]

        # load data
        optical = Image.open(self.optical_path, fname).convert('RGB') # 0-255
        sar = Image.open(self.sar_path, fname).convert('L') # 0-255
        label = Image.open(self.label_path, fname)

        # convert to tensor

        # transforms

        return optical, sar, label

def get_dataloader(config):
    data = OpenEarthMapSarDataset(config.dataset_root)
    indices = np.arange(len(data))

    train_index, temp_index = train_test_split(indices, train_size=config.train_size, random_state=42, shuffle=True)
    val_index, test_index = train_test_split(indices, train_size=config.val_size, random_state=42, shuffle=True)

    train_loader = DataLoader(Subset(data, train_index), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(data, val_index), batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(data, test_index), batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_index