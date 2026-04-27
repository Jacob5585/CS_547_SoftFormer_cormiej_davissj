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

### Addations:
- Dataloader
- Patch_embedding
- Config
- Data download scripts
- Evaluation
- Checkpoints
- Unifed pipeline for differnt model methods

### Modifications:
- Dataset -> Semantic Segmentation dataset with 8 classes since the orginal dataset isn't available
- Segmentation -> BRIEF description of modifications

### Run:
#### Train
```
python train.py --epochs <number of epochs> --method <classification or segmentation> --batch-size <image batch size> --dataset-root <path to the root folder of the dataset> --checkpoint_path <path to checkpoint if being used> 
```

#### Test
```
```


# <font color=#5dbe8a>SoftFormer</font>
## Intorduction
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


