import numpy as np
from sklearn.metrics import confusion_matrix

class Evaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))
    
    def update(self, labels, preds):
        mask = (labels >= 0) & (labels < self.num_classes)
        self.confusion_matrix += confusion_matrix(labels[mask].flatten(), preds[mask].flatten(), labels=range(self.num_classes))

    def compute_metrics(self):
        cm = self.confusion_matrix
        oa = np.trace(cm) / np.sum(cm)

        # kappa
        total = np.sum(cm)
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (total**2)
        kappa = (oa - pe) / (1 - pe + 1e-10)

        # mIoU
        intersection = np.diag(cm)
        union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - intersection
        iou = intersection / (union + 1e-10)
        miou = np.nanmean(iou)
    
        # F1 Score
        precision = intersection / (np.sum(cm, axis=0) + 1e-10)
        recall = intersection / (np.sum(cm, axis=1) + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        mf1 = np.nanmean(f1)
        
        return {"OA": oa, "Kappa": kappa, "mIoU": miou, "mF1": mf1, "Class_F1": f1}

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))