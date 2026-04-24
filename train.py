from config import Config
import dataloader
import evaluation
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from Network import SoftFormer

scaler = torch.amp.GradScaler('cuda')

def train(config):
    results = []
    start_epoch = 0
    train_loader, val_loader, _ = dataloader.get_dataloader(config)

    # choose image size based on method
    img_size = config.patch_size if config.method == "classification" else config.image_size
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

    if config.checkpoint_path and os.path.exists(config.checkpoint_path):
        print(f"Loading checkpoint: {config.checkpoint_path}")
        checkpoint = torch.load(config.checkpoint_path, map_location=config.device, weights_only=False)
        
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

    for epoch in range(start_epoch, config.epochs):
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
        result.update(metrics)
        
        if isinstance(result['Class_F1'], np.ndarray):
            result['Class_F1'] = result['Class_F1'].tolist()

        result = {}
        results.append(result)
        
        with open(f'metrics_{config.method}_train.jsonl', 'a') as file:
            file.write(json.dumps(result) + '\n')

        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "metrics": metrics}
        torch.save(checkpoint, os.path.join(config.save_directory, f"model_{config.method}_{epoch}.pth"))
        
        if metrics['mIoU'] > best_miou:
            best_miou = metrics['mIoU']
            torch.save(checkpoint, os.path.join(config.save_directory, f"model_{config.method}_{epoch}_best_mIoU.pth"))

    metrics = evaluator.compute_metrics()
    checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "metrics": metrics}
    torch.save(checkpoint, os.path.join(config.save_directory, "model.pth"))

    print(results)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SoftFormer model.")
    parser.add_argument("--dataset-root", default=None, help="Override dataset root from config.")
    parser.add_argument("--method", default=None, choices=["classification", "segmentation"], help="Model method to evaluate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config.")
    parser.add_argument("--checkpoint_path", default=None, help="Checkpoint to load from")
    args = parser.parse_args()

    config = Config()
    if args.dataset_root is not None:
        config.dataset_root = args.dataset_root
    if args.method is not None:
        config.method = args.method
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.checkpoint_path is not None:
        config.checkpoint_path = args.checkpoint_path

    train(config)

if __name__ == "__main__":
    main()