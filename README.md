# This is a fork created for a CS 547 class project at SUNY Polytechnic
## Authors: Jacob Cormier Sawyer Davis

### Setup:
```
git clone https://github.com/Jacob5585/CS_547_SoftFormer_cormiej_davissj.git
cd CS_547_SoftFormer_cormiej_davissj
conda create --name softformer python=3.12
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 #other version of cuda will work such as https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Download dataset with the download.sh for linux and mac or the download.ps1 for windows. Or go `https://zenodo.org/records/14622048/files/dfc25_track1_trainval.zip?download=1` in your browser

### Additions:
- Dataloader
- Patch_embedding
- Config
- Data download scripts
- Evaluation
- Checkpoints
- Unifed pipeline for differnt model methods

### Modifications:
- Dataset -> Semantic Segmentation dataset with 8 classes since the orginal dataset isn't available
- Segmentation -> Added `Network_seg.py` containing `SoftFormerSeg`, a segmentation variant of the original classification network. The dual-branch encoder (optical + SAR) and feature-level fusion (`FeatFuseBlock`) are reused unchanged. The classification heads (`Classifier`, `DecisionFuseBlock`) are replaced with a lightweight `SegmentationHead` decoder that progressively upsamples the fused 2×2 feature map back to a higher resolution via two bilinear upsample + conv stages, producing per-pixel class logits instead of a single class score per patch.

### Run:
#### Train
```
python train.py --epochs <number of epochs> --method <classification or segmentation> --batch-size <image batch size> --dataset-root <path to the root folder of the dataset> --checkpoint_path <path to checkpoint if being used> 
```

#### Test
```
python test.py <path/to/checkpoint.pth> [--method classification|segmentation] [--dataset-root /path/to/data] [--batch-size 32] [--save-preds] [--save-dir test_outputs]
```

Examples:
```bash
# Train segmentation model from scratch
python train.py --epochs 50 --method segmentation --batch-size 16 --dataset-root /data/mydata

# Train classification model from scratch
python train.py --epochs 50 --method classification --batch-size 32 --dataset-root /data/mydata

# Resume training segmentation model from checkpoint
python train.py --epochs 100 --method segmentation --batch-size 16 --dataset-root /data/mydata --checkpoint_path checkpoints/seg_best.pth
```

Examples:
```bash
# Evaluate segmentation model
python test.py checkpoints/seg_best.pth --method segmentation

# Evaluate classification model and save predictions
python test.py checkpoints/cls_best.pth --method classification --save-preds

# Override dataset root and batch size
python test.py checkpoints/seg_best.pth --method segmentation --dataset-root /data/mydata --batch-size 16
```

The script prints per-metric results to stdout. If `--save-preds` is set, flattened predictions are saved as `<save-dir>/preds.npy`.

---

# <font color=#5dbe8a>SoftFormer</font>
## Introduction
SoftFormer is a deep learning network designed for urban land cover and land use classification based on SAR and Optical data fusion. The rationale behind SoftFormer is:<br> 

(1) Feature Extraction: balancing the local and global feature extraction capabilities of the network.<br>

(2) Feature Fusion: enhancing the classification accuracy of heterogenous remote sensing data by multi-level (feature level and decision level) fusion. When fusing, the complementarity and redundancy of heterogeneous source features need to be considered, as well as the issue of the credibility of the classification results of different modalities.<br>

SoftFormer is a patch based **classification** network rather than **segmentation** network. When the labeled samples are prepared by ENVI or GEE, you can use SoftFormer to do classification.

## Citations
If this work is helpful to you, please consider citing the paper via the following BibTex entry.
```bibtex
@article{LIU2024277,
title = {SoftFormer: SAR-optical fusion transformer for urban land use and land cover classification},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {218},
pages = {277-293},
year = {2024},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2024.09.012},
author = {Rui Liu and Jing Ling and Hongsheng Zhang},
}
```