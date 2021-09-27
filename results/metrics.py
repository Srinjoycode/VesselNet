import torch
import numpy as np
import torchmetrics
import torchmetrics.functional
import pandas as pd

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


def check_metrics(loader, model, writer, device="cuda"):
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
    
    print(adding_metrics(epoch_no, float(acc), float(total_iou_score),
                                        float(batch_dice_score),
                                        float(total_f1),
                                        float(total_precision),
                                        float(total_recall),
                                        float(total_specificity)))
    
    preds = preds.numpy()
    y = y.numpy()
    preds = preds.ravel()
    y = y.ravel()
    
    
    roc_curve_plot(y, preds)
    precision_recall_curve_plot(y, preds)


    model.train()  # end of model evaluation
    
    

    
#function to upadate dataframe which contains all the mertics for each epoch  
prediction = pd.DataFrame(columns=['Epoch_no','Accuracy','IoU','Dice','f1_score','Precision','Recall','Specificity'])
def adding_metrics(epoch_no, accuracy, iou,dice, f1_score, precision, recall, specificity):
    global prediction
    new_row = {'Epoch_no': epoch_no, 
               'Accuracy': accuracy, 
               'IoU': iou,
               'Dice': dice,
               'f1_score': f1_score,
               'Precision': precision,
               'Recall': recall,
               'Specificity':specificity}
    prediction = prediction.append(new_row, ignore_index=True)
    return prediction


#plotting roc  curve for each epoch
from sklearn import datasets, metrics
import matplotlib.pyplot as plt
def roc_curve_plot(y_true, y_preds):
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_preds)
    print(fpr)
    print(tpr)
    print(thresholds)
    AUC_ROC = metrics.roc_auc_score(y_true, y_preds)
    # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
    print ("\nArea under the ROC curve: " +str(AUC_ROC))
    roc_curve =plt.figure()
    plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")


#plotting presicion-recall for each epoch
def precision_recall_curve_plot(y_true, y_preds):
    
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_preds)
    precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision,recall)
    print ("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
    prec_rec_curve = plt.figure()
    plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    
    
#plotting curve for all the metrics
import matplotlib.pyplot as plt
import seaborn as sns
def plotting_metrics(file_name):
    
    #accuracy
    fig, ax = plt.subplots(figsize=(8, 8))
    file_name['Accuracy'].plot.line(color = 'red')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.show()
    plt.savefig('accuracy.png', dpi=300, bbox_inches='tight')
    
    #IOU
    fig, ax = plt.subplots(figsize=(8, 8))
    file_name['IoU'].plot.line(color = 'blue')
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("IoU")
    plt.show()
    plt.savefig('IoU.png', dpi=300, bbox_inches='tight')
   
    #Dice
    fig, ax = plt.subplots(figsize=(8, 8))
    file_name['Dice'].plot.line(color = 'green')
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Dice")
    plt.show()
    plt.savefig('dice.png', dpi=300, bbox_inches='tight')
    
    #f1_score
    fig, ax = plt.subplots(figsize=(8, 8))
    file_name['f1_score'].plot.line(color = 'blue')
    plt.xlabel("Epoch")
    plt.ylabel("f1_score")
    plt.title("f1_score")
    plt.show()
    plt.savefig('f1_score.png', dpi=300, bbox_inches='tight')
    
    #Precision
    fig, ax = plt.subplots(figsize=(8, 8))
    file_name['Precision'].plot.line(color = 'orange')
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Precision")
    plt.show()
    plt.savefig('precision.png', dpi=300, bbox_inches='tight')
    
    #Recall
    fig, ax = plt.subplots(figsize=(8, 8))
    file_name['Recall'].plot.line(color = 'purple')
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Recall")
    plt.show()
    plt.savefig('recall.png', dpi=300, bbox_inches='tight')
    
    #Specificity
    fig, ax = plt.subplots(figsize=(8, 8))
    file_name['Specificity'].plot.line(color = 'red')
    plt.xlabel("Epoch")
    plt.ylabel("Specificity")
    plt.title("Specificity")
    plt.show()
    plt.savefig('specificity.png', dpi=300, bbox_inches='tight')

#function to convert it to csv file
def convert_to_csv(prediction):
    prediction.to_csv(r'./metrics.csv', header=True)

