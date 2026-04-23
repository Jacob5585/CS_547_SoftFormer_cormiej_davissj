from config import Config
import dataloader
import evaluation
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from Network import SoftFormer

scaler = torch.amp.GradScaler('cuda')

CHECKPOINT_PATH = "checkpoints.model_18_best_mIoU.pth"

def train():
    results = []
    config = Config()
    train_loader, val_loader, _ = dataloader.get_dataloader(config)

    model = SoftFormer(
            opt_chans=config.optical_channels,
            sar_chans=config.sar_channels,
            num_class=config.num_classes,
            img_size=config.patch_size if config.method == "classification" else config.image_size
        ).to(config.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    evaluator = evaluation.Evaluator(config.num_classes)

    # Mean Intersection over Union (mIoU) 
    best_miou = 0.0

    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=config.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Resume from the next epoch
        start_epoch = checkpoint['epoch'] + 1
        
        # Restore best_miou if it exists in the saved metrics
        if 'metrics' in checkpoint:
            best_miou = checkpoint['metrics'].get('mIoU', 0.0)
            
        # Update scheduler to match the current epoch
        for _ in range(start_epoch):
            scheduler.step()
            
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for opt, sar, label in progress_bar:
            opt, sar, label = opt.to(config.device), sar.to(config.device), label.to(config.device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
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
        result = {
            'epoch': epoch + 1,
            'OA': metrics['OA'],
            'mIoU': metrics['mIoU'],
            'Kappa': metrics['Kappa']
        }
        results.append(result)
        
        with open('metrics_train.json', 'a') as file:
            json.dump(result, file, indent=4)

        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "metrics": metrics}
        torch.save(checkpoint, os.path.join(config.save_directory, f"model_{epoch}.pth"))
        
        if metrics['mIoU'] > best_miou:
            best_miou = metrics['mIoU']
            torch.save(checkpoint, os.path.join(config.save_directory, f"model_{epoch}_best_mIoU.pth"))

    metrics = evaluator.compute_metrics()
    checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "metrics": metrics}
    torch.save(checkpoint, os.path.join(config.save_directory, "model.pth"))

    print(results)

if __name__ == "__main__":
    train()