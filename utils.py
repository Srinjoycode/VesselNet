import torch
import torchvision

from torch.utils.data import DataLoader
from Datasets.dataloader import IMG_Dataset


#TODO add a dataloader for DRIVE dataset
def get_loaders(
        train_dir,
        train_mask_dir,
        val_dir,
        val_mask_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True
):
    train_dataset = IMG_Dataset(
        image_dir=train_dir,
        mask_dir=train_mask_dir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_dataset = IMG_Dataset(
        image_dir=val_dir,
        mask_dir=val_mask_dir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    return train_loader, val_loader


def save_checkpoint(state, filename="Vessel_Net_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, epoch, loss):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred/{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/truth_labels/{idx}.png")
    model.train()