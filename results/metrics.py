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
import threading
import time
from results.roc_precall import roc_curve_plot, precision_recall_curve_plot


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

def make_csv_copy(metrics_dir,prev_metrics_csv_dir):
    metrics_df_old = pd.read_csv(prev_metrics_csv_dir, index_col=False)
    convert_to_csv(metrics_df_old,metrics_dir)



def check_metrics(train_loader, val_loader, model, epoch_no, last_epoch,
                  loss_fn, train_loss, load_model, device="cuda",
                  metrics_dir='./new_metrics.csv',
                  prev_metrics_csv_dir='./prev_metrics.csv'):
    global adding_m
    adding_m = dict()

    s = time.perf_counter()

    def training_train(epoch_no, last_epoch):
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
                preds = torch.sigmoid(model(x))
                predictions = model(x)
                adding_m['train_loss'] = loss_fn(predictions, y).item()
                predictions = torch.sigmoid(predictions)

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

                if(last_epoch == True):
                    predictions = predictions.cpu().numpy().ravel()
                    y = y.numpy().ravel()
                    roc_curve_plot(y, predictions, "ROC Curve - Training", epoch_no)
                    precision_recall_curve_plot(y, predictions, "Precision Recall Curve - Training", epoch_no)

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

        adding_m['train_accuracy'] = train_acc.detach().cpu().numpy()
        adding_m['train_iou'] = (train_total_iou_score.detach().cpu().numpy() / len(train_loader))
        adding_m['train_dice'] = (train_batch_dice_score.detach().cpu().numpy() / len(train_loader))
        adding_m['train_f1_score'] = (train_total_f1.detach().cpu().numpy() / len(train_loader))
        adding_m['train_precision'] = (train_total_precision.detach().cpu().numpy() / len(train_loader))
        adding_m['train_recall'] = (train_total_recall.detach().cpu().numpy() / len(train_loader))
        adding_m['train_specificity'] = (train_total_specificity.detach().cpu().numpy() / len(train_loader))
        adding_m['train_mcc'] = (train_total_mcc.detach().cpu().numpy() / len(train_loader))


    def validation_train(epoch_no, last_epoch):

        #validation
        global metrics_dir_path
        metrics_dir_path = metrics_dir
        batch_num_correct = 0
        batch_num_pixels = 0
        val_batch_dice_score = 0
        val_total_iou_score = 0
        val_total_precision = 0
        val_total_recall = 0
        val_total_f1 = 0
        val_total_specificity = 0
        val_total_mcc = 0

        if load_model == True:
            if (not os.path.isfile(metrics_dir)):
                make_csv_copy(metrics_dir, prev_metrics_csv_dir)

        # start of model evaluation
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)  # storing the input image
                y = y.to(device).unsqueeze(1)  # storing the original mask
                preds = torch.sigmoid(model(x))
                predictions = model(x)
                adding_m['val_loss'] = loss_fn(predictions, y).item()
                predictions = torch.sigmoid(predictions)

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

                if(last_epoch == True):
                    predictions = predictions.cpu().numpy().ravel()
                    y = y.numpy().ravel()
                    roc_curve_plot(y, predictions, "ROC Curve - Validation", epoch_no)
                    precision_recall_curve_plot(y, predictions, "Precision Recall Curve - Validation", epoch_no)

        val_acc = accuracy_score(batch_num_correct, batch_num_pixels)

        print('VALIDATION SCORES')
        print(f"Got {batch_num_correct}/{batch_num_pixels} with val_acc {val_acc}")

        print(f"mean IoU score: {val_total_iou_score / len(val_loader)}")

        print(f"mean Dice Score: {val_batch_dice_score / len(val_loader)}")

        print(f"torch-metrics mean f1_score: {val_total_f1 / len(val_loader)}")

        print(f"torch-metrics mean precision: {val_total_precision / len(val_loader)}")

        print(f"torch-metrics mean recall: {val_total_recall / len(val_loader)}")

        print(f"torch-metrics mean specificity: {val_total_specificity / len(val_loader)}")

        print(f"Custom MCC metrics value:{val_total_mcc / len(val_loader)}")

        adding_m['val_accuracy'] = val_acc.detach().cpu().numpy()
        adding_m['val_iou'] = (val_total_iou_score.detach().cpu().numpy() / len(val_loader))
        adding_m['val_dice'] = (val_batch_dice_score.detach().cpu().numpy() / len(val_loader))
        adding_m['val_f1_score'] = (val_total_f1.detach().cpu().numpy() / len(val_loader))
        adding_m['val_precision'] = (val_total_precision.detach().cpu().numpy() / len(val_loader))
        adding_m['val_recall'] = (val_total_recall.detach().cpu().numpy() / len(val_loader))
        adding_m['val_specificity'] = (val_total_specificity.detach().cpu().numpy() / len(val_loader))
        adding_m['val_mcc'] = (val_total_mcc.detach().cpu().numpy() / len(val_loader))

    train_thread = threading.Thread(target=training_train, args=(epoch_no, last_epoch, ))
    validation_thread = threading.Thread(target=validation_train, args=(epoch_no, last_epoch, ))
    train_thread.start()
    validation_thread.start()
    train_return = train_thread.join()
    validation_return = validation_thread.join()

    print(adding_metrics(epoch_no = int(epoch_no),

                         val_accuracy = adding_m['val_accuracy'],
                         val_iou = adding_m['val_iou'],
                         val_dice = adding_m['val_dice'],
                         val_f1_score = adding_m['val_f1_score'],
                         val_precision = adding_m['val_precision'],
                         val_recall = adding_m['val_recall'],
                         val_specificity = adding_m['val_specificity'],
                         val_mcc = adding_m['val_mcc'],

                         train_accuracy = adding_m['train_accuracy'],
                         train_iou = adding_m['train_iou'],
                         train_dice = adding_m['train_dice'],
                         train_f1_score = adding_m['train_f1_score'],
                         train_precision = adding_m['train_precision'],
                         train_recall = adding_m['train_recall'],
                         train_specificity = adding_m['train_specificity'],
                         train_mcc = adding_m['train_mcc'],

                         train_loss= adding_m['train_loss'],
                         val_loss= adding_m['val_loss'],

                         load_model = load_model,
                         metrics_dir=metrics_dir,
                         )
          )
    # # Plotting the ROC and Precision vs recall curves
    # preds = preds.numpy().ravel()
    # y = y.numpy().ravel()
    Results_dataframe = pd.read_csv(metrics_dir, index_col=False)
    #
    # np.savetxt("preds.csv", preds, delimiter=",")
    # np.savetxt("true.csv", y, delimiter=",")

    f = time.perf_counter()
    print(f'Metrics calculation finished in {f - s} second(s) for epoch: {epoch_no}')

    model.train()  # end of model evaluation

# function to update dataframe which contains all the metrics for each epoch
prediction = pd.DataFrame(
    columns=['Epoch_no', 'Val_Accuracy', 'Val_IoU', 'Val_Dice', 'Val_f1_score', 'Val_Precision', 'Val_Recall', 'Val_Specificity', 'Train_Accuracy', 'Train_IoU', 'Train_Dice', 'Train_f1_scorVal_e', 'Train_Precision', 'Train_Recall', 'Train_Specificity'])

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


# function to convert it to csv file
def convert_to_csv(prediction,metrics_dir):
    prediction.to_csv(metrics_dir, header=True, index=False)

if __name__ == "__main__":
    # Test Metric Plotting

    Results_dataframe = pd.read_csv(metrics_dir_path, index_col=False)
    # plotting_metrics(Results_dataframe)
    #plot_loss(Results_dataframe)
