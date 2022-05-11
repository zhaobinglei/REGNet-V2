import torch
import torch.nn as nn
import torch.nn.functional as F

from tdgpd.functions.gather_knn import gather_knn
from tdgpd.nn.modules import *


class EdgeConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.bn = nn.BatchNorm2d(2 * out_channels)

    def forward(self, feature, knn_inds):
        batch_size, _, num_points = feature.shape
        k = knn_inds.shape[2]

        local_feature = self.conv1(feature)  # (batch_size, out_channels, num_points)
        edge_feature = self.conv2(feature)  # (batch_size, out_channels, num_points)
        channels = local_feature.shape[1]

        # custom improved gather
        neighbour_feature = gather_knn(edge_feature, knn_inds)

        central_feature = local_feature.unsqueeze(-1).expand(-1, -1, -1, k)

        edge_feature = torch.cat([central_feature, neighbour_feature - central_feature], dim=1)

        edge_feature = self.bn(edge_feature)
        edge_feature = F.relu(edge_feature, inplace=True)
        edge_feature = torch.mean(edge_feature, dim=3)

        return edge_feature


class EdgeConvNoC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeConvNoC, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, feature, knn_inds):
        batch_size, _, num_points = feature.shape
        k = knn_inds.shape[2]

        local_feature = self.conv1(feature)  # (batch_size, out_channels, num_points)
        edge_feature = self.conv2(feature)  # (batch_size, out_channels, num_points)
        channels = local_feature.shape[1]

        if feature.is_cuda:
            # custom improved gather
            neighbour_feature = gather_knn(edge_feature, knn_inds)
        else:
            # pytorch gather
            knn_inds_expand = knn_inds.unsqueeze(1).expand(batch_size, channels, num_points, k)
            edge_feature_expand = edge_feature.unsqueeze(2).expand(batch_size, -1, num_points, num_points)
            neighbour_feature = torch.gather(edge_feature_expand, 3, knn_inds_expand)

        # (batch_size, out_channels, num_points, k)
        central_feature = local_feature.unsqueeze(-1).expand(-1, -1, -1, k)

        edge_feature = neighbour_feature - central_feature
        edge_feature = self.bn(edge_feature)
        edge_feature = F.relu(edge_feature, inplace=True)
        edge_feature = torch.mean(edge_feature, dim=3)

        return edge_feature
