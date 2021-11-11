import torch
import numpy as np
import torchmetrics
import torchmetrics.functional
import pandas as pd
from sklearn import datasets, metrics
import matplotlib.pyplot as plt
import seaborn as sns
import os

step = 0


def dice_score(preds, y):
    return (2 * (preds * y).sum()) / (preds + y).sum() + 1e-8


def num_correct_pixels(preds, y):
    return (preds == y).sum()


def num_total_pixels(preds):
    return torch.numel(preds)


def accuracy_score(num_correct, num_pixels):
    return num_correct / num_pixels * 100


def iou(preds, y):
    preds = preds.type(torch.int)
    y = y.type(torch.int)
    intersection = np.logical_and(y, preds)
    union = np.logical_or(y, preds)
    iou_score = intersection.sum() / union.sum()
    return iou_score


def precision(preds, y):
    return torchmetrics.functional.precision(preds, y, num_classes=2, mdmc_average='global')


def recall(preds, y):
    return torchmetrics.functional.recall(preds, y, num_classes=2, mdmc_average='global')


def f1(preds, y):
    return torchmetrics.functional.f1(preds, y, num_classes=2, mdmc_average='global')


def specificity(preds, y):
    return torchmetrics.functional.specificity(preds, y, num_classes=2, mdmc_average='global')


def mcc(preds, y):
    return torchmetrics.functional.matthews_corrcoef(preds, y, num_classes=2)


def check_metrics(loader, model, writer, epoch_no, last_epoch, load_model, device="cuda", ):
    global step

    # global variables definitions
    batch_num_correct = 0
    batch_num_pixels = 0
    batch_dice_score = 0
    total_iou_score = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_specificity = 0
    total_mcc = 0
    # start of model evaluation
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)  # storing the input image
            y = y.to(device).unsqueeze(1)  # storing the original mask
            # converting the predictions to a binary map
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            batch_num_pixels += num_total_pixels(preds)
            y, preds = y.cpu(), preds.cpu()

            batch_num_correct += num_correct_pixels(preds, y)
            batch_dice_score += dice_score(preds, y)
            total_iou_score += iou(preds, y)

            preds = preds.type(torch.int)
            y = y.type(torch.int)
            total_f1 += f1(preds, y)
            total_precision += precision(preds, y)
            total_recall += recall(preds, y)
            total_specificity += specificity(preds, y)
            total_mcc += mcc(preds, y)

    acc = accuracy_score(batch_num_correct, batch_num_pixels)
    writer["writer"].add_scalar("Training Accuracy", acc, global_step=step)
    step += 1

    print(f"Got {batch_num_correct}/{batch_num_pixels} with acc {acc}")

    print(f"mean IoU score: {total_iou_score / len(loader)}")

    print(f"mean Dice Score: {batch_dice_score / len(loader)}")

    print(f"torch-metrics mean f1_score: {total_f1 / len(loader)}")

    print(f"torch-metrics mean precision: {total_precision / len(loader)}")

    print(f"torch-metrics mean recall: {total_recall / len(loader)}")

    print(f"torch-metrics mean specificity: {total_specificity / len(loader)}")

    print(f"Custom MCC metrics value:{total_mcc / len(loader)}")

    print(adding_metrics(int(epoch_no), acc.detach().cpu().numpy(),
                         (total_iou_score.detach().cpu().numpy() / len(loader)),
                         (batch_dice_score.detach().cpu().numpy() / len(loader)),
                         (total_f1.detach().cpu().numpy() / len(loader)),
                         (total_precision.detach().cpu().numpy() / len(loader)),
                         (total_recall.detach().cpu().numpy() / len(loader)),
                         (total_specificity.detach().cpu().numpy() / len(loader)),
                         (total_mcc.detach().cpu().numpy() / len(loader)),
                         load_model
                         )
          )

    # TODO CHECK FOR LAST EPOCH AND OPTIMIZE

    # Plotting the ROC and Precision vs recall curves
    preds = preds.numpy().ravel()
    y = y.numpy().ravel()
    Results_dataframe = pd.read_csv('./metrics.csv', index_col=False)
    plotting_metrics(Results_dataframe)
    roc_curve_plot(y, preds)
    precision_recall_curve_plot(y, preds)


    model.train()  # end of model evaluation


# function to update dataframe which contains all the metrics for each epoch
prediction = pd.DataFrame(
    columns=['Epoch_no', 'Accuracy', 'IoU', 'Dice', 'f1_score', 'Precision', 'Recall', 'Specificity'])


def adding_metrics(epoch_no, accuracy, iou, dice, f1_score, precision, recall, specificity, mcc, load_model):

    global prediction
    if bool(load_model):
        prediction = pd.read_csv('./metrics.csv', index_col=False)

    new_row = {'Epoch_no': epoch_no,
               'Accuracy': accuracy,
               'IoU': iou,
               'Dice': dice,
               'f1_score': f1_score,
               'Precision': precision,
               'Recall': recall,
               'Specificity': specificity,
               'MCC': mcc,
               }
    prediction = prediction.append(new_row, ignore_index=True)
    convert_to_csv(prediction)
    return prediction


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
    sns.lineplot(x='Epoch_no',y='Accuracy',data=Results_dataframe,color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Graph")

    plt.savefig('./metrics_plots/accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

    # IOU
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(x='Epoch_no',y='IoU',data=Results_dataframe,color='green')
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("IoU Score")

    plt.savefig('./metrics_plots/IoU.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Dice
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(x='Epoch_no',y='Dice',data=Results_dataframe,color='black')
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Dice")

    plt.savefig('./metrics_plots/dice.png', dpi=300, bbox_inches='tight')
    plt.close()

    # f1_score
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(x='Epoch_no',y='f1_score',data=Results_dataframe,color='darkkhaki')
    plt.xlabel("Epoch")
    plt.ylabel("F1 score")
    plt.title("F1 score")

    plt.savefig('./metrics_plots/f1_score.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Precision
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(x='Epoch_no',y='Precision',data=Results_dataframe,color='purple')
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Precision")

    plt.savefig('./metrics_plots/precision.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Recall
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(x='Epoch_no',y='Recall',data=Results_dataframe,color='brown')
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Recall")

    plt.savefig('./metrics_plots/recall.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Specificity
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(x='Epoch_no',y='Specificity',data=Results_dataframe,color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Specificity")
    plt.title("Specificity")

    plt.savefig('./metrics_plots/specificity.png', dpi=300, bbox_inches='tight')
    plt.close()

    # MCC
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(x='Epoch_no', y='MCC', data=Results_dataframe,color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("MCC")
    plt.title("MCC Score")

    plt.savefig('./metrics_plots/MCC.png', dpi=300, bbox_inches='tight')
    plt.close()


# function to convert it to csv file
def convert_to_csv(prediction):
    prediction.to_csv(r'./metrics.csv', header=True, index=False)
