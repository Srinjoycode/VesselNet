import os
import argparse
import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from models.VesselNet import Vessel_net
from results.metrics import check_metrics
from utils import load_checkpoint, save_checkpoint, get_loaders, save_predictions_as_imgs

torch.cuda.empty_cache()

step = 0


# USIng Mixed precision training (FP-16 used )
def train_fn(loader, model, optimizer, loss_fn, scaler, args, writer):
    global step
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=args.device)
        targets = targets.float().unsqueeze(1).to(device=args.device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        img_grid = torchvision.utils.make_grid(data)
        writer.add_image("Input_image", img_grid)
        # writer.add_histogram("fc1", model.fc1.weight)
        writer.add_scalar("Training Loss", loss, global_step=step)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        step += 1


def main(args):
    global step
    torch.cuda.empty_cache()
    train_transform = A.Compose(
        [
            A.Resize(height=args.height, width=args.width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=args.height, width=args.width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    model = Vessel_net(in_channels=3, out_channels=1).to(args.device)

    train_loader, val_loader = get_loaders(
        args.train_dir,
        args.train_mask,
        args.val_dir,
        args.val_mask,
        args.batch_size,
        train_transform,
        val_transforms,
        args.num_workers,
        args.pin_memory,
    )

    # since no sigmoid on the output of the model we use with logits
    loss_fn = nn.BCEWithLogitsLoss()  # for multiclass classification use cross entropy loss and change  #out_channels
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(
        f"runs/Dataset/Minibatch {args.batch_size} LR {args.lr}"
    )

    images, _ = next(iter(train_loader))
    writer.add_graph(model, images.to(args.device))
    writer.close()
    PREV_EPOCHS = 0
    if args.load_model:
        PREV_EPOCHS = load_checkpoint(torch.load(
            args.load_weights),
            model)
        print("Loaded Model Metrics")
        #print("Epoch Metrics are being printed for epoch num :"+str(PREV_EPOCHS))
        #check_metrics(loader=val_loader, model=model, device=args.device,epoch_no=PREV_EPOCHS, writer={"writer": writer, "step": step},last_epoch= False,load_model= args.load_model)


    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(PREV_EPOCHS+1,args.epochs+1):
        print("Epoch : " + str(epoch))

        train_fn(train_loader, model, optimizer, loss_fn, scaler, args, writer=writer)

        # save model and predictions
        if epoch == (args.epochs - 1):
            last_epoch = True
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss_fn,
            }
            # TODO Add args for saving checkpoint script file path
            checkpoint_name = './VesselNetChase_Epochs' + str(epoch) + '_CHASE.pth.tar'
            save_checkpoint(checkpoint, checkpoint_name)
            try:
                os.mkdir('validation_saved_images')
                os.mkdir('validation_saved_images/pred')
                os.mkdir('validation_saved_images/truth_labels')
            except:
                print("Results directory already created")
                pass
            save_predictions_as_imgs(val_loader, model, folder="validation_saved_images", device=args.device)
        else:
            last_epoch = False
        #CHECK METRICS
        print("Epoch Metrics are being printed for epoch num :" + str(epoch))
        check_metrics(val_loader, model, device=args.device, epoch_no=int(epoch),writer={"writer": writer, "step": step},last_epoch=last_epoch,load_model= bool(args.load_model) )

        step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning Rate for training")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch Size for training")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--epochs", default=10, type=int, help="Number of Epochs")
    parser.add_argument("--num_workers", default=2, help="Number of workers")
    parser.add_argument("--height", default=256, type=int, help="Input Image Height")
    parser.add_argument("--width", default=512, type=int, help="Input Image width")
    parser.add_argument("--pin_memory", default=True, help="Pin Memory")
    parser.add_argument("--load_model",type=bool, default=False, help="Load Pretrained Model")
    parser.add_argument("--train_dir", default="Datasets/CHASE/train/images", type=str,
                        help="Training images directory")
    parser.add_argument("--train_mask", default="Datasets/CHASE/train/labels", type=str,
                        help="Training mask directory")
    parser.add_argument("--val_dir", default="Datasets/CHASE/validate/images",
                        help="Validation Image directory")
    parser.add_argument("--val_mask", type=str, default="Datasets/CHASE/validate/labels",
                        help="Validation label directory")
    parser.add_argument("--test_dir", default="Datasets/CHASE/test/images", type=str,
                        help="Test image directory")
    parser.add_argument("--test_mask", default="Datasets/CHASE/test/labels",
                        help="Test mask directory")
    parser.add_argument("--load_weights", default="trained.pth.tar", type=str, help="Add training weight path")
    args = parser.parse_args()

    main(args)