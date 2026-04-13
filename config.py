import os
import torch

class Config:
    dataset_root = os.path.join("dataset", "openearthmap-sar", "train")
    save_directory = "checkpoints"

    num_classes = 8
    image_size = 1024
    train_size = 0.7
    val_size = 0.1
    test_size = 0.2

    # hyperparamters
    epochs = 10
    batch_size = 8
    learning_rate = 3e-3
    weight_decay = 1e-4

    optical_channels = 3
    sar_channels = 1
    patch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = ["Bareland", "Rangeland", "Developed Space", "Road", "Tree", "Water", "Agricultural Land", "Building"]