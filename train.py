from config import Config
import dataloader
import evaluation
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from Network import SoftFormer

scaler = torch.cuda.amp.GradScaler()

def train():
    result = []
    config = Config()
    train_loader, val_loader, _ = dataloader.get_dataloader(config)

    model = SoftFormer(
            opt_chans=config.optical_channels,
            sar_chans=config.sar_channels,
            num_class=config.num_classes,
            img_size=config.patch_size if config.method == "classification" elif config.method == "segmentation" config.image_size
        ).to(config.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    evaluator = evaluation.Evaluator(config.num_classes)

    # Mean Intersection over Union (mIoU) 
    best_miou = 0.0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for opt, sar, label in progress_bar:
            opt, sar, label = opt.to(config.device), sar.to(config.device), label.to(config.device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                outputs = model(opt, sar)
                loss = criterion(outputs[0], label)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        scheduler.step()

        model.eval()
        evaluator.reset()
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]")
            
        with torch.no_grad():
            for i, (opt, sar, label) in enumerate(val_progress_bar):
                opt, sar = opt.to(config.device), sar.to(config.device)
                val_outputs = model(opt, sar)
                preds = val_outputs[0].argmax(dim=1)
                   
                evaluator.update(label.cpu().numpy(), preds.cpu().numpy())

        metrics = evaluator.compute_metrics()
        result.append({
            'epoch': epoch + 1,
            'OA': metrics['OA'],
            'mIoU': metrics['mIoU'],
            'Kappa': metrics['Kappa']
        })

        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "metrics": metrics}
        torch.save(checkpoint, os.path.join(config.save_directory, f"model_{epoch}.pth"))
        
        if metrics['mIoU'] > best_miou:
            best_miou = metrics['mIoU']
            torch.save(checkpoint, os.path.join(config.save_directory, f"model_{epoch}_best_mIoU.pth"))

    metrics = evaluator.compute_metrics()
    checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "metrics": metrics}
    torch.save(checkpoint, os.path.join(config.save_directory, "model.pth"))

    print(result)

if __name__ == "__main__":
    train()