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

def roc_curve_plot(y_true, y_preds):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_preds)
    print(fpr)
    print(tpr)
    print(thresholds)
    AUC_ROC = metrics.roc_auc_score(y_true, y_preds)

    print("\nArea under the ROC curve: " + str(AUC_ROC))
    roc_curve = plt.figure()
    plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC, )
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")

    plt.savefig('./metrics_plots/roc_curve_plot.png', dpi=300, bbox_inches='tight')
    plt.close()


# plotting presicion-recall for each epoch
def precision_recall_curve_plot(y_true, y_preds):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_preds)
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision, recall)
    print("\nArea under Precision-Recall curve: " + str(AUC_prec_rec))
    prec_rec_curve = plt.figure()
    plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig('./metrics_plots/precision_recall_curve_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

# plotting curve for all the metrics
def plotting_metrics(Results_dataframe):
    try:
        os.mkdir('./metrics_plots')

    except:
        print(" Metrics Plots directory already created")
        pass

    # accuracy
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(x='Epoch_no',y='Train_Accuracy',data=Results_dataframe,color='red')
    sns.lineplot(x='Epoch_no',y='Val_Accuracy',data=Results_dataframe,color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Graph")
    plt.legend(["Train", "Val"], loc = "upper right")

    plt.savefig('./metrics_plots/accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

    # IOU
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(x='Epoch_no',y='Train_IoU',data=Results_dataframe,color='red')
    sns.lineplot(x='Epoch_no',y='Val_IoU',data=Results_dataframe,color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("IoU Score")
    plt.legend(["Train", "Val"], loc = "upper right")

    plt.savefig('./metrics_plots/IoU.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Dice
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(x='Epoch_no',y='Train_Dice',data=Results_dataframe,color='red')
    sns.lineplot(x='Epoch_no',y='Val_Dice',data=Results_dataframe,color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Dice")
    plt.legend(["Train", "Val"], loc = "upper right")

    plt.savefig('./metrics_plots/dice.png', dpi=300, bbox_inches='tight')
    plt.close()

    # f1_score
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(x='Epoch_no',y='Train_f1_score',data=Results_dataframe,color='red')
    sns.lineplot(x='Epoch_no',y='Val_f1_score',data=Results_dataframe,color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("F1 score")
    plt.title("F1 score")
    plt.legend(["Train", "Val"], loc = "upper right")

    plt.savefig('./metrics_plots/f1_score.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Precision
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(x='Epoch_no',y='Train_Precision',data=Results_dataframe,color='red')
    sns.lineplot(x='Epoch_no',y='Val_Precision',data=Results_dataframe,color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Precision")
    plt.legend(["Train", "Val"], loc = "upper right")

    plt.savefig('./metrics_plots/precision.png', dpi=300, bbox_inches='tight')
    plt.close()
    plt.legend(["Train", "Val"], loc = "upper right")

    # Recall
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(x='Epoch_no',y='Train_Recall',data=Results_dataframe,color='red')
    sns.lineplot(x='Epoch_no',y='Val_Recall',data=Results_dataframe,color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Recall")
    plt.legend(["Train", "Val"], loc = "upper right")

    plt.savefig('./metrics_plots/recall.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Specificity
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(x='Epoch_no',y='Train_Specificity',data=Results_dataframe,color='red')
    sns.lineplot(x='Epoch_no',y='Val_Specificity',data=Results_dataframe,color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Specificity")
    plt.title("Specificity")
    plt.legend(["Train", "Val"], loc = "upper right")

    plt.savefig('./metrics_plots/specificity.png', dpi=300, bbox_inches='tight')
    plt.close()

    # MCC
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(x='Epoch_no', y='Train_MCC', data=Results_dataframe,color='red')
    sns.lineplot(x='Epoch_no', y='Val_MCC', data=Results_dataframe,color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("MCC")
    plt.title("MCC Score")
    plt.legend(["Train", "Val"], loc = "upper right")

    plt.savefig('./metrics_plots/MCC.png', dpi=300, bbox_inches='tight')
    plt.close()

# Plot Training and Validation loss vs EPOCHS
def plot_loss(dataframe):
    # IOU
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(x='Epoch_no',y='Train_loss',data=dataframe,color='red')
    sns.lineplot(x='Epoch_no',y='Val_loss',data=dataframe,color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LOSS vs EPOCH")
    plt.legend(["Train", "Val"], loc = "upper right")
    plt.savefig('./metrics_plots/loss.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    Results_dataframe = pd.read_csv("new_metrics.csv", index_col = False)
    # preds = np.loadtxt("preds.csv")
    # y = np.loadtxt("true.csv")

    plotting_metrics(Results_dataframe)
    plot_loss(Results_dataframe)
    # roc_curve_plot(y, preds)
    # precision_recall_curve_plot(y, preds)