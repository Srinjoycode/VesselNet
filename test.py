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

class IMG_test_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg', '_predicted_mask.jpg'))
        image = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)



        if self.transform is  not None:
            augmentations = self.transform(image = image)
            image = augmentations["image"]


        return image,mask_path

def test(args):
    images, masks, weights, device = args.input_dir, args.mask_dir, args.weights, args.device

    if not os.path.exists("test_metrics"):
        os.makedirs("test_metrics")

    # Dataloader

    print("Loading Inference Images.....")

    dataset = IMG_test_Dataset(images,masks)

    print("Images Loaded")


    print("Loading model")
    model = Vessel_net()
    model.to(device)
    print("Model loaded")



    print("Loading trained model parameters")
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Loaded model parameters")

    print("Saving model as ONNX Format for interoperability")
    dummy_input = torch.randn(args.batch_size, 3, args.height_onnx_model,args.width_onnx_model, requires_grad=True )
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model,
                      dummy_input,
                      args.onnx_model_name,
                      verbose=False,
                      input_names=input_names,
                      output_names=output_names,
                      do_constant_folding=True,
                      export_params=True,
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}}
                      )




    for img_path,mask_path in dataset:
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
    parser.add_argument("--batch_size", default=1, type=int, help="Batch Size for testing")
    parser.add_argument("--input_dir", type=str, required=True, help="Relative path to test image path")
    parser.add_argument("--mask_dir", type=str, required=True, help="Relative path to test masks(initially blank)")
    parser.add_argument("--weights", default="trained.pth.tar", type=str,
                        help="Relative path to the trained model path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--onnx_model_name", default="VesselNet.onnx", help="name given to the model")
    parser.add_argument("--height_onnx_model", type=int,default=512, help="Height ")
    parser.add_argument("--width_onnx_model", type=int ,default=512, help="cuda or cpu")
    args = parser.parse_args()

    test(args)