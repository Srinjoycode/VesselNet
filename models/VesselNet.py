# Import the libraries
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from models.conv_attention import CBAMBlock
from models.conv_blocks import DoubleConv, SingleConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Defining the main Encoder decoder model using the double convolution class
class Vessel_net(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[16, 32, 64]):
        super(Vessel_net, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.singleconv1 = SingleConv(in_channels=features[-1], out_channels=features[-1] * 2)
        self.singleconv2 = SingleConv(in_channels=features[-1] * 2, out_channels=features[-1] * 2)

        # encoder part (down part)
        for feature in features:
            self.downs.append(DoubleConv(in_channels=in_channels, out_channels=feature))
            in_channels = feature

        # decoder part (up part)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(in_channels=feature * 2,
                                   out_channels=feature,
                                   kernel_size=2,
                                   stride=2
                                   )
            )
            self.ups.append(
                DoubleConv(in_channels=feature * 2, out_channels=feature)
            )

        self.bottleneck = CBAMBlock(channel=256) #CHANGE THIS WHEN BIG MODEL IS USED
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.singleconv1(x)
        x = self.bottleneck(x)
        x = self.singleconv2(x)
        # print('size before entering the decoder ' + str(x.size()))
        # reverse the skip connection list
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        x = self.final_conv(x)
        return x


def test():
    x = torch.randn((1, 3, 584, 584)).to(device)
    model = Vessel_net(in_channels=3, out_channels=1).to(device)
    preds = model(x).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(model)
    print("Shape of the Input image: " + str(x.shape))
    print("shape of the output segmentation map: " + str(preds.shape))
    print("Number of Model Parameters: ", total_params)


if __name__ == '__main__':
    test()
