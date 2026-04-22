print('test.py started')
print('test.py is running')
import argparse
import os
import numpy as np
import torch
from config import Config
import dataloader
import evaluation
from Network import SoftFormer as SoftFormerCls
from Network_seg import SoftFormerSeg


def load_checkpoint(model, path, device):
    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as err:
        if 'weights_only' in str(err) or 'Unsupported global' in str(err):
            with torch.serialization.safe_globals([np._core.multiarray.scalar]):
                checkpoint = torch.load(path, map_location=device, weights_only=False)
        else:
            raise
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    return model


def evaluate_model(model, loader, device, num_classes, method):
    evaluator = evaluation.Evaluator(num_classes)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for opt, sar, label in loader:
            opt = opt.to(device)
            sar = sar.to(device)
            outputs = model(opt, sar)
            logits = outputs[0] if method == "classification" else outputs
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(label.numpy())
            evaluator.update(label.numpy(), preds)

    all_preds = np.concatenate([p.reshape(-1) for p in all_preds])
    all_labels = np.concatenate([l.reshape(-1) for l in all_labels])
    metrics = evaluator.compute_metrics()
    return all_preds, all_labels, metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SoftFormer model.")
    parser.add_argument("model_path", help="Path to the saved model checkpoint.")
    parser.add_argument("--dataset-root", default=None, help="Override dataset root from config.")
    parser.add_argument("--method", default=None, choices=["classification", "segmentation"], help="Model method to evaluate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config.")
    parser.add_argument("--save-preds", action="store_true", help="Save predicted labels to a numpy file.")
    parser.add_argument("--save-dir", default="test_outputs", help="Directory to save prediction outputs.")
    args = parser.parse_args()

    config = Config()
    if args.dataset_root:
        config.dataset_root = args.dataset_root
    if args.method:
        config.method = args.method
    if args.batch_size:
        config.batch_size = args.batch_size

    _, _, test_loader = dataloader.get_dataloader(config)

    model_cls = SoftFormerCls if config.method == "classification" else SoftFormerSeg
    img_size = config.patch_size if config.method == "classification" else config.image_size
    model = model_cls(
        opt_chans=config.optical_channels,
        sar_chans=config.sar_channels,
        num_class=config.num_classes,
        img_size=img_size,
    ).to(config.device)

    load_checkpoint(model, args.model_path, config.device)

    preds, labels, metrics = evaluate_model(model, test_loader, config.device, config.num_classes, config.method)

    print("Test metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    if args.save_preds:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, "preds.npy")
        np.save(save_path, preds)
        print(f"Saved flattened predictions to {save_path}")

    return metrics


if __name__ == "__main__":
    main()
