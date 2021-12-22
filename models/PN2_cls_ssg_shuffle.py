import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ShufflePointNet_util import PointNetSetShuffleAbstraction, FeatureRecolor


# from torchsummary import summary


class get_model(nn.Module):
    def __init__(self, num_class, conv_groups=2, normal_channel=True, **kwargs):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.conv_groups = conv_groups
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetShuffleAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel,
                                                 mlp=[64, 64, 128], conv_groups=self.conv_groups, group_all=False,
                                                 global_shuffle=True)
        self.sa2 = PointNetSetShuffleAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3,
                                                 mlp=[128, 128, 256], conv_groups=self.conv_groups, group_all=False,
                                                 global_shuffle=True)
        self.sa3 = PointNetSetShuffleAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                                 mlp=[256, 512, 1024], conv_groups=self.conv_groups, group_all=True,
                                                 global_shuffle=True)
        self.fr1 = FeatureRecolor(128)
        self.fr2 = FeatureRecolor(256)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.fr1(self.sa1(xyz, norm))
        l2_xyz, l2_points = self.fr2(self.sa2(l1_xyz, l1_points))
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, model: torch.nn.Module):
        total_loss = F.nll_loss(pred, target)
        return total_loss


if __name__ == '__main__':
    # model = get_model(40, 2, normal_channel=True).cuda()
    # input = torch.randn(24, 6, 1024).cuda()
    # summary(model, input)
    import time

    model = get_model(40, normal_channel=False).cuda()
    input = torch.randn(24, 3, 1024).cuda()
    model(input)
    iterations = 100
    ttt = []
    with torch.no_grad():
        for i in range(iterations):
            t1 = time.time()
            model(input)
            t2 = time.time()
            ttt.append(t2 - t1)
    print(min(ttt) / 24)

    # macs, params, time = profile(model, inputs=(input,))
    # print(clever_format([macs, params], "%.3f"), time / 64)
