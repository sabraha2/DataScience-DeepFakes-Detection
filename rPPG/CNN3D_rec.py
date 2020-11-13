import torch
import torch.nn as nn
import torch.nn.functional as F


class HRCNN(nn.Module):
    def __init__(self, channels=3, drop_p=0.2, t_kern=7):
        super(HRCNN, self).__init__()

        t_pad = int(t_kern / 2)

        self.conv1 = nn.Conv3d(in_channels=channels, out_channels=32, kernel_size=(1,5,5), padding=(0,2,2))
        self.bn1 = nn.BatchNorm3d(32)
        self.max_pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(t_kern,3,3), padding=(t_pad,1,1))
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(t_kern,3,3), padding=(t_pad,1,1))
        self.bn3 = nn.BatchNorm3d(64)
        self.max_pool2 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(t_kern,3,3), padding=(t_pad,1,1))
        self.bn4 = nn.BatchNorm3d(64)

        self.conv5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(t_kern,3,3), padding=(t_pad,1,1))
        self.bn5 = nn.BatchNorm3d(64)
        self.max_pool3 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv6 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(t_kern,3,3), padding=(t_pad,1,1))
        self.bn6 = nn.BatchNorm3d(64)

        self.conv7 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(t_kern,3,3), padding=(t_pad,1,1))
        self.bn7 = nn.BatchNorm3d(64)
        self.max_pool4 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv8 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(t_kern,3,3), padding=(t_pad,1,1))
        self.bn8 = nn.BatchNorm3d(64)

        self.conv9 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(t_kern,3,3), padding=(t_pad,1,1))
        self.bn9 = nn.BatchNorm3d(64)

        self.avg_pool1 = nn.AvgPool3d(kernel_size=(1,4,4), stride=(1,2,2))
        self.conv10 = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=1)

        self.drop3d = nn.Dropout3d(drop_p)

        self.forward_stream = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(), self.max_pool1,
            self.conv2, self.bn2, nn.ReLU(),
            self.conv3, self.bn3, nn.ReLU(), self.drop3d, self.max_pool2,
            self.conv4, self.bn4, nn.ReLU(),
            self.conv5, self.bn5, nn.ReLU(), self.drop3d, self.max_pool3,
            self.conv6, self.bn6, nn.ReLU(),
            self.conv7, self.bn7, nn.ReLU(), self.drop3d, self.max_pool4,
            self.conv8, self.bn8, nn.ReLU(),
            self.conv9, self.bn9, nn.ReLU(), self.drop3d, self.avg_pool1,
            self.conv10
        )


    def forward(self, x):
        x = self.forward_stream(x)
        x = torch.flatten(x, start_dim=1, end_dim=4)
        return x
