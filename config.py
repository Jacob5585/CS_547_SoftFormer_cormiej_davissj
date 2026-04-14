import os
import torch

class Config:
    dataset_root = os.path.join("dataset", "openearthmap-sar", "train")
    save_directory = "checkpoints"

    class_names = ["Bareland", "Rangeland", "Developed Space", "Road", "Tree", "Water", "Agricultural Land", "Building"]
    num_classes = len(class_names)
    image_size = 8 #256
    train_size = 0.7
    val_size = 0.1
    test_size = 0.2

    # hyperparamters
    epochs = 10
    batch_size = 1024
    learning_rate = 3e-3
    weight_decay = 1e-4

    optical_channels = 3
    sar_channels = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
