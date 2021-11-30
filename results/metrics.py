import torch
import numpy as np
import torchmetrics
import torchmetrics.functional
import pandas as pd
from sklearn import datasets, metrics
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch._C import _valgrind_toggle

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


def check_metrics(train_loader, val_loader, model,writer , epoch_no, last_epoch, loss_fn, train_loss, load_model, device="cuda",metrics_dir='./metrics.csv' ):
    global step
    global metrics_dir_path
    metrics_dir_path = metrics_dir

    # global variables definitions
    batch_num_correct = 0
    batch_num_pixels = 0
    val_batch_dice_score = 0
    val_total_iou_score = 0
    val_total_precision = 0
    val_total_recall = 0
    val_total_f1 = 0
    val_total_specificity = 0
    val_total_mcc = 0
    # start of model evaluation
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)  # storing the input image
            y = y.to(device).unsqueeze(1)  # storing the original mask
            # converting the predictions to a binary map
            preds = torch.sigmoid(model(x))
            predictions = model(x)
            val_loss = loss_fn(predictions, y).item()

            batch_num_pixels += num_total_pixels((preds > 0.5).float())
            y, preds = y.cpu(), (preds > 0.5).float().cpu()

            batch_num_correct += num_correct_pixels((preds > 0.5).float(), y)
            val_batch_dice_score += dice_score((preds > 0.5).float(), y)
            val_total_iou_score += iou((preds > 0.5).float(), y)


            y = y.type(torch.int)
            val_total_f1 += f1((preds > 0.5).float().type(torch.int), y)
            val_total_precision += precision((preds > 0.5).float().type(torch.int), y)
            val_total_recall += recall((preds > 0.5).float().type(torch.int), y)
            val_total_specificity += specificity((preds > 0.5).float().type(torch.int), y)
            val_total_mcc += mcc((preds > 0.5).float().type(torch.int), y)

    val_acc = accuracy_score(batch_num_correct, batch_num_pixels)
    writer["writer"].add_scalar("Training Accuracy", val_acc, global_step=step)
    step += 1
    print('VALIDATION SCORES')
    print(f"Got {batch_num_correct}/{batch_num_pixels} with val_acc {val_acc}")

    print(f"mean IoU score: {val_total_iou_score / len(val_loader)}")

    print(f"mean Dice Score: {val_batch_dice_score / len(val_loader)}")

    print(f"torch-metrics mean f1_score: {val_total_f1 / len(val_loader)}")

    print(f"torch-metrics mean precision: {val_total_precision / len(val_loader)}")

    print(f"torch-metrics mean recall: {val_total_recall / len(val_loader)}")

    print(f"torch-metrics mean specificity: {val_total_specificity / len(val_loader)}")

    print(f"Custom MCC metrics value:{val_total_mcc / len(val_loader)}")

    # global variables definitions
    batch_num_correct = 0
    batch_num_pixels = 0
    train_batch_dice_score = 0
    train_total_iou_score = 0
    train_total_precision = 0
    train_total_recall = 0
    train_total_f1 = 0
    train_total_specificity = 0
    train_total_mcc = 0
    # start of model evaluation
    model.eval()
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)  # storing the input image
            y = y.to(device).unsqueeze(1)  # storing the original mask
            # converting the predictions to a binary map
            preds = torch.sigmoid(model(x))
            predictions = model(x)
            # train_loss = loss_fn(predictions, y).item()

            batch_num_pixels += num_total_pixels((preds > 0.5).float())
            y, preds = y.cpu(), (preds > 0.5).float().cpu()

            batch_num_correct += num_correct_pixels((preds > 0.5).float(), y)
            train_batch_dice_score += dice_score((preds > 0.5).float(), y)
            train_total_iou_score += iou((preds > 0.5).float(), y)

            y = y.type(torch.int)
            train_total_f1 += f1((preds > 0.5).float().type(torch.int), y)
            train_total_precision += precision((preds > 0.5).float().type(torch.int), y)
            train_total_recall += recall((preds > 0.5).float().type(torch.int), y)
            train_total_specificity += specificity((preds > 0.5).float().type(torch.int), y)
            train_total_mcc += mcc((preds > 0.5).float().type(torch.int), y)

    train_acc = accuracy_score(batch_num_correct, batch_num_pixels)

    print('TRAINING SCORES')

    print(f"Got {batch_num_correct}/{batch_num_pixels} with train_acc {train_acc}")

    print(f"mean IoU score: {train_total_iou_score / len(train_loader)}")

    print(f"mean Dice Score: {train_batch_dice_score / len(train_loader)}")

    print(f"torch-metrics mean f1_score: {train_total_f1 / len(train_loader)}")

    print(f"torch-metrics mean precision: {train_total_precision / len(train_loader)}")

    print(f"torch-metrics mean recall: {train_total_recall / len(train_loader)}")

    print(f"torch-metrics mean specificity: {train_total_specificity / len(train_loader)}")

    print(f"Custom MCC metrics value:{train_total_mcc / len(train_loader)}")

    # Adding Metrics to CSV
    print(adding_metrics(epoch_no=int(epoch_no),

                         val_accuracy=val_acc.detach().cpu().numpy(),
                         val_iou=(val_total_iou_score.detach().cpu().numpy() / len(val_loader)),
                         val_dice=(val_batch_dice_score.detach().cpu().numpy() / len(val_loader)),
                         val_f1_score=(val_total_f1.detach().cpu().numpy() / len(val_loader)),
                         val_precision=(val_total_precision.detach().cpu().numpy() / len(val_loader)),
                         val_recall=(val_total_recall.detach().cpu().numpy() / len(val_loader)),
                         val_specificity=(val_total_specificity.detach().cpu().numpy() / len(val_loader)),
                         val_mcc=(val_total_mcc.detach().cpu().numpy() / len(val_loader)),

                         train_accuracy=train_acc.detach().cpu().numpy(),
                         train_iou=(train_total_iou_score.detach().cpu().numpy() / len(train_loader)),
                         train_dice=(train_batch_dice_score.detach().cpu().numpy() / len(train_loader)),
                         train_f1_score=(train_total_f1.detach().cpu().numpy() / len(train_loader)),
                         train_precision=(train_total_precision.detach().cpu().numpy() / len(train_loader)),
                         train_recall=(train_total_recall.detach().cpu().numpy() / len(train_loader)),
                         train_specificity=(train_total_specificity.detach().cpu().numpy() / len(train_loader)),
                         train_mcc=(train_total_mcc.detach().cpu().numpy() / len(train_loader)),

                         train_loss=train_loss,
                         val_loss=val_loss,

                         load_model=load_model,
                        metrics_dir=metrics_dir,
                         )
          )

    # Plotting the ROC and Precision vs recall curves
    preds = preds.numpy().ravel()
    y = y.numpy().ravel()
    Results_dataframe = pd.read_csv(metrics_dir, index_col=False)
    plotting_metrics(Results_dataframe)
    plot_loss(Results_dataframe)
    roc_curve_plot(y, preds)
    precision_recall_curve_plot(y, preds)
    model.train()  # end of model evaluation


# function to update dataframe which contains all the metrics for each epoch
prediction = pd.DataFrame(
    columns=['Epoch_no', 'Val_Accuracy', 'Val_IoU', 'Val_Dice', 'Val_f1_score', 'Val_Precision', 'Val_Recall',
             'Val_Specificity', 'Train_Accuracy', 'Train_IoU', 'Train_Dice', 'Train_f1_scorVal_e', 'Train_Precision',
             'Train_Recall', 'Train_Specificity'])

# For adding Metircs to CSV

def adding_metrics(epoch_no, train_accuracy, val_accuracy, train_iou, val_iou, train_dice, val_dice, train_f1_score, val_f1_score, train_precision, val_precision, train_recall, val_recall, train_specificity, val_specificity, train_mcc, val_mcc, train_loss, val_loss, load_model,metrics_dir):

    global prediction
    if bool(load_model):
        prediction = pd.read_csv(metrics_dir, index_col=False)

    new_row = {'Epoch_no': epoch_no,

               'Val_Accuracy': val_accuracy,
               'Val_IoU': val_iou,
               'Val_Dice': val_dice,
               'Val_f1_score': val_f1_score,
               'Val_Precision': val_precision,
               'Val_Recall': val_recall,
               'Val_Specificity': val_specificity,
               'Val_MCC': val_mcc,

               'Train_Accuracy': train_accuracy,
               'Train_IoU': train_iou,
               'Train_Dice': train_dice,
               'Train_f1_score': train_f1_score,
               'Train_Precision': train_precision,
               'Train_Recall': train_recall,
               'Train_Specificity': train_specificity,
               'Train_MCC': train_mcc,

               'Train_loss': train_loss,
               'Val_loss':val_loss,
               }
    prediction = prediction.append(new_row, ignore_index=True)
    convert_to_csv(prediction,metrics_dir)
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

# function to convert it to csv file
def convert_to_csv(prediction,metrics_dir):
    prediction.to_csv(metrics_dir, header=True, index=False)

if __name__ == "__main__":
    # Test Metric Plotting

    Results_dataframe = pd.read_csv(metrics_dir_path, index_col=False)
    # plotting_metrics(Results_dataframe)
    plot_loss(Results_dataframe)
