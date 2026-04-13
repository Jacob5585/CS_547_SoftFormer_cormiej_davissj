from config import Config
import dataloader
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from Network import SoftFormer

def train():
    config = Config()
    train_loader, val_loader = dataloader.get_dataloader(config)

    model = SoftFormer(num_class=config.number_classes).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss()
    # evaluator = Evaluator(config.num_classes)

    # Mean Intersection over Union (mIoU) 
    best_miou = 0.0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for opt, sar, label in progress_bar:
            opt, sar, label = opt.to(config.device), sar.to(config.device), label.to(config.device)

            optimizer.zero_grad()
            outputs = model(opt, sar)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

        scheduler.step()

        if epoch % 5 == 0:
            model.eval()
            # evalator.reset()
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]")
            
            with torch.no_grad():
                for opt, sar, label in val_progress_bar:
                    opt, sar = opt.to(config.device), sar.to(config.device)
                    preds = model(opt, sar).argmax(dim=1)
                    # evaluator.update(label.numpy(), preds.cpu().numpy())
            
            # metrics = evaluator.compute_metrics()
            print(f"")

            checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "metrics": metrics}
            torch.save(checkpoint, os.path.join(config.save_directory, "model.pth"))

    
    checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "metrics": metrics}
    torch.save(checkpoint, os.path.join(config.save_directory, "model.pth"))


if __name__ == "__main__":
    train()