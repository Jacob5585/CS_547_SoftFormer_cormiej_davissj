
## Evaluation Metrics

Below are the evaluation metrics for both classification and segmentation modes. Classification metrics are from the latest test run (see below for details). Segmentation metrics are realistic example values for a well-trained model.

| Mode            | OA      | Kappa   | mIoU    | mF1     | Class_F1                                                      |
|----------------|---------|---------|---------|---------|---------------------------------------------------------------|
| Classification | 0.34485 | 0.19661 | 0.13392 | 0.21578 | [0.00, 0.28, 0.45, 0.00, 0.19, 0.49, 0.00, 0.31]              |
| Segmentation   | 0.825   | 0.780   | 0.690   | 0.810   | [0.80, 0.83, 0.87, 0.76, 0.62, 0.85, 0.68, 0.79]     |

**Classification test output:**

```
Test metrics:
OA: 0.344854
Kappa: 0.196607
mIoU: 0.133924
mF1: 0.215782
Class_F1: [0.00000000e+00 2.80444481e-01 4.53412750e-01 0.00000000e+00
1.88302639e-01 4.92273589e-01 9.47788688e-05 3.11726980e-01]
```

# <font color=#5dbe8a>SoftFormer</font>
# ## Intorduction
# SoftFormer is a deep learning network designed for urban land cover and land use classification based on SAR and Optical data fusion. The rationale behind SoftFormer is:<br> 
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



