import argparse
import time
import os
import torchvision
import torch
import numpy as np
from torch.utils.data import Dataset

from models.VesselNet import Vessel_net
from PIL import Image

#TODO DRIVE testloader and test script

class ChaseTestLoader(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg', '_1stHO.png'))
        image = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask, mask_path


def test(args):
    images, masks, weights, device = args.input_dir, args.mask_dir, args.weights, args.device

    if not os.path.exists("test_metrics"):
        os.makedirs("test_metrics")

    # Dataloader

    print("Loading Inference Images.....")

    dataset = ChaseTestLoader(images,masks)

    print("images Loaded")


    print("Loading model")
    model = Vessel_net()
    model.to(device)
    print("Model loaded")

    print("Loading trained model parameters")
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Loaded model parameters")




    for img_path,mask,mask_path in dataset:
        img = torch.from_numpy(img_path).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        img = img.permute(0,3,1,2)
        print(img.size())
        t1 = time.time()
        prediction = torch.sigmoid(model(img))
        prediction = (prediction > 0.5).float()
        prediction = prediction.squeeze()
        print(prediction.shape)
        print("MODEL PREDICTION TIME: ", time.time() - t1)
        out = prediction.detach().cpu().numpy()
        print("Output shape"+str(out.shape))

        torchvision.utils.save_image(prediction, f"{mask_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Relative path to test image path")
    parser.add_argument("--mask_dir", type=str, required=True, help="Relative path to test masks(initially blank)")
    parser.add_argument("--weights", default="trained.pth.tar", type=str,
                        help="Relative path to the trained model path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    test(args)