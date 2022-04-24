import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation = 1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, dilation=dilation),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, dilation=dilation),
            nn.Dropout2d(p=0.2, inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )
    def forward(self, x):
        return self.conv(x)


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation = 1):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, dilation=dilation),
            nn.Dropout2d(p=0.2, inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )
    def forward(self, x):
        x=self.conv(x)
        return x