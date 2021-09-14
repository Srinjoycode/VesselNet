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

#function to convert it to csv file
def convert_to_csv(prediction):
    prediction.to_csv(r'./metrics.csv', header=True)

