import torch
from torch import nn as nn
from torch.utils.checkpoint import checkpoint as ckpt

from models.pointnet_util import sample_and_group_all, sample_and_group, square_distance, index_points


class FeatureRecolor(nn.Module):
    def __init__(self, in_channel, downsample_factor=4):
        super(FeatureRecolor, self).__init__()
        self.gapool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(in_channel + 3, in_channel // downsample_factor, 1)
        self.fc2 = nn.Conv1d(in_channel // downsample_factor, in_channel, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xyz_, points = x
        feat = torch.cat([xyz_, points], dim=1)
        feat = self.gapool(feat)
        feat = self.fc1(feat)
        feat = self.relu(feat)
        feat = self.fc2(feat)
        feat = self.sigmoid(feat)
        output = feat * points.clone()
        output = output + points
        return xyz_, output


class PointShuffleUnit(nn.Module):
    def __init__(self, in_channel, out_channel, conv_group=2, global_shuffle=False, info_training=False):
        super(PointShuffleUnit, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv_groups = conv_group
        self.global_shuffle = global_shuffle
        self.info = info_training
        if in_channel == 3 or in_channel % 2 != 0 or in_channel == 6:
            self.conv_groups = 1
        self.conv_branch_1 = nn.Sequential(
            nn.Conv2d(self.in_channel // 2, self.in_channel // 2, 1, bias=False),
            nn.BatchNorm2d(self.in_channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channel // 2, self.in_channel // 2, 1, bias=False, groups=self.conv_groups),
            nn.BatchNorm2d(self.in_channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channel // 2, self.in_channel // 2, 1, bias=False),
            nn.BatchNorm2d(self.in_channel // 2),
            nn.ReLU(inplace=True),
        )
        if self.in_channel != self.out_channel:
            self.conv_branch_2 = nn.Sequential(
                nn.Conv2d(self.in_channel, self.out_channel, 1, bias=False, groups=1),
                nn.BatchNorm2d(self.out_channel),
                nn.ReLU(inplace=True),
            )
        if self.info:
            self.localdis = LocalDiscriminator(self.out_channel)

    @staticmethod
    @torch.jit.script
    def _shuffle(points, groups: int, global_shuffle: bool):
        tmp = points.clone()

        B, C, S, P = points.data.size()
        samples_pre_group = S // groups
        points = points.view(B, C, groups, samples_pre_group, P)
        points = torch.transpose(points, 2, 3).contiguous()
        points = points.view(B, C, -1, P)
        if global_shuffle:
            points = points.view(B, groups, C // groups, S, P)
            points = torch.transpose(points, 1, 2).contiguous()
            points = points.view(B, -1, S, P)
        return points, tmp

    def shuffle(self, points, groups: int = 2):
        # B, C, S, P = points.data.size()
        # samples_pre_group = S // groups
        # points = points.view(B, C, groups, samples_pre_group, P)
        # points = torch.transpose(points, 2, 3).contiguous()
        # points = points.view(B, C, -1, P)
        # if self.global_shuffle:
        #     points = points.view(B, groups, C // groups, S, P)
        #     points = torch.transpose(points, 1, 2).contiguous()
        #     points = points.view(B, -1, S, P)

        points, tmp = self._shuffle(points, groups, self.global_shuffle)

        # if self.info:
        #     shuffle_score = self.localdis(torch.cat([tmp, tmp], dim=1))
        #     unshuffle_score = self.localdis(torch.cat([tmp, points], dim=1))
        #     del tmp
        #     local_info_loss = - torch.mean(torch.log(unshuffle_score + 1e-6) + torch.log(1 - shuffle_score + 1e-6))
        #     del shuffle_score, unshuffle_score
        #     self.register_buffer("local_info_loss", local_info_loss)

        return points

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.conv_branch_1(x2)
        out = torch.cat((x1, x2), dim=1)
        out = out + x
        if self.in_channel != self.out_channel:
            out = self.conv_branch_2(out)
        out = self.shuffle(out)

        return out


class PointShuffleUnit1D(nn.Module):
    def __init__(self, in_channel, out_channel, conv_groups=4, global_shuffle=True, attention=False, is_compact=False):
        super(PointShuffleUnit1D, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv_groups = conv_groups
        self.global_shuffle = global_shuffle
        self.attention = attention
        if in_channel % 2 != 0 or in_channel % 4 != 0:
            self.conv_groups = 1
        if is_compact:
            self.conv_branch_1 = nn.Sequential(
                nn.Conv1d(self.in_channel // 2, self.in_channel // 2, 1, bias=False, groups=self.conv_groups),
                nn.BatchNorm1d(self.in_channel // 2),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv_branch_1 = nn.Sequential(
                nn.Conv1d(self.in_channel // 2, self.in_channel // 2, 1, bias=False),
                nn.BatchNorm1d(self.in_channel // 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.in_channel // 2, self.in_channel // 2, 1, bias=False, groups=self.conv_groups),
                nn.BatchNorm1d(self.in_channel // 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.in_channel // 2, self.in_channel // 2, 1, bias=False),
                nn.BatchNorm1d(self.in_channel // 2),
                nn.ReLU(inplace=True)
            )
        if self.in_channel != self.out_channel:
            if is_compact:
                self.conv_branch_2 = nn.Sequential(
                    nn.Conv1d(self.in_channel, self.out_channel, 1, bias=False, groups=self.conv_groups),
                    nn.BatchNorm1d(self.out_channel),
                    nn.ReLU(inplace=True),
                )
            else:
                self.conv_branch_2 = nn.Sequential(
                    nn.Conv1d(self.in_channel, self.out_channel, 1, bias=False, groups=self.conv_groups),
                    nn.BatchNorm1d(self.out_channel),
                    nn.ReLU(inplace=True),
                )

    def shuffle(self, points, groups=2):
        B, C, P = points.shape

        points = points.view(B, groups, C // groups, P)
        points = torch.transpose(points, 1, 2).contiguous()
        points = points.view(B, -1, P)

        return points

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.conv_branch_1(x2)
        out = torch.cat((x1, x2), dim=1)
        out = out + x
        if self.in_channel != self.out_channel:
            out = self.conv_branch_2(out)
        out = self.shuffle(out, 2)

        return out


class PointNetSetShuffleAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all,
                 conv_groups=2, global_shuffle=False, is_compact=False, checkpoint=False, info=False, use_cluster=False,
                 **kwargs):
        super(PointNetSetShuffleAbstraction, self).__init__()
        self.use_cluster = use_cluster
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.conv_groups = conv_groups
        self.global_shuffle = global_shuffle
        self.mlp_convs = nn.ModuleList()
        self.checkpoint = checkpoint
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(PointShuffleUnit(last_channel, out_channel, conv_group=self.conv_groups,
                                                   global_shuffle=self.global_shuffle, info_training=info))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            data = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, use_cluster=self.use_cluster)
            if len(data) == 2:
                new_xyz, new_points = data
            else:
                new_xyz, new_points, _ = data
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            if self.checkpoint:
                new_points = ckpt(conv, new_points)
            else:
                new_points = conv(new_points)

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetFeatureShufflePropagation(nn.Module):
    def __init__(self, in_channel, mlp, conv_groups=4, is_compact=False, checkpoint=False):
        super(PointNetFeatureShufflePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.conv_groups = conv_groups
        self.checkpoint = checkpoint
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(
                PointShuffleUnit1D(last_channel, out_channel, conv_groups=self.conv_groups, is_compact=is_compact))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            if self.checkpoint:
                new_points = conv(new_points)
            else:
                new_points = ckpt(conv, new_points)
        return new_points


class LocalDiscriminator(nn.Module):
    def __init__(self, channel):
        super(LocalDiscriminator, self).__init__()
        self.l1 = nn.Conv2d(channel * 2, channel // 4, kernel_size=1)
        self.l2 = nn.Conv2d(channel // 4, 1, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.sigmoid(self.l2(x))

        return torch.flatten(x)
