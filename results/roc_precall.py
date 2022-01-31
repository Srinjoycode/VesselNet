import torch
import numpy as np
from torch._C import _valgrind_toggle
import torchmetrics
import torchmetrics.functional
import pandas as pd
from sklearn import datasets, metrics
import matplotlib.pyplot as plt
import seaborn as sns
import os

# plotting roc  curve for each epoch

def roc_curve_plot(y_true, y_preds, title, epoch_no, location):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_preds)
    # print(fpr)
    # print(tpr)
    # print(thresholds)
    AUC_ROC = metrics.roc_auc_score(y_true, y_preds)

    print("\nArea under the ROC curve: " + str(AUC_ROC))
    roc_curve = plt.figure()
    plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC, )
    plt.title(title)
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    if(title == "ROC Curve - Training"):
        plt.savefig(f'{location}/training_roc_curve_plot_{epoch_no}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'{location}/validation_roc_curve_plot_{epoch_no}.png', dpi=300, bbox_inches='tight')
    plt.close()


# plotting presicion-recall for each epoch
def precision_recall_curve_plot(y_true, y_preds, title, epoch_no, location):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_preds)
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision, recall)
    print("\nArea under Precision-Recall curve: " + str(AUC_prec_rec))
    prec_rec_curve = plt.figure()
    plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    if(title == "Precision Recall Curve - Training"):
        plt.savefig(f'{location}/training_precision_recall_curve_plot_{epoch_no}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'{location}/validation_precision_recall_curve_plot_{epoch_no}.png', dpi=300, bbox_inches='tight')
    plt.close()
