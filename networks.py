import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, dilation=2, padding=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, dilation=3, padding=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, dilation=4, padding=4)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.norm = nn.LayerNorm([64, 256, 256])

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6,
                      self.conv7, self.conv8, self.conv9, self.conv10, self.conv11, self.conv12]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, image):
        x1 = F.silu(self.conv1(image))
        x1 = F.silu(self.conv2(x1))

        x2 = F.silu(self.conv3(image))
        x2 = F.silu(self.conv4(x2))

        x1_x2 = torch.cat([x1, x2], dim=1)
        x3 = F.silu(self.conv5(x1_x2))
        add1 = image + x3

        x4 = F.silu(self.conv6(add1))
        x4 = self.conv7(x4)
        add2 = x4 + add1
        add2 = F.silu(add2)

        x5 = F.silu(self.conv8(add2))
        x5 = F.silu(self.conv9(x5))
        x5 = self.conv10(x5)

        add3 = add2 + x5
        add3 = F.silu(add3)

        gap = F.adaptive_avg_pool2d(add3, (1, 1))
        x6 = F.silu(self.conv11(gap))
        x6 = torch.sigmoid(self.conv12(x6))

        mul = x6 * add3
        output = image + mul

        return output


class RIDNet(nn.Module):
    def __init__(self):
        super(RIDNet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.eam = nn.Sequential(
            EAM(),
            EAM(),
            EAM(),
            EAM()
        )
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=3, padding=1)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, image):
        x = self.conv1(image)
        x = self.eam(x)
        x = self.conv2(x)

        output = image + x

        return output
    
if __name__ == "__main__":
    
    model_path = "../Project/saved_models/ridnet3_epoch_16patch29oct.pth"
    checkpoint = torch.load(model_path)
    model = RIDNet()
    summary(model, (3, 256, 256), device="cpu")