import torch
import torch.nn as nn
from source.semantic_segmentation.models.model_utils import get_graph_feature


class DGCNNSemSeg(nn.Module):
    def __init__(self, num_classes: int, k: int, embedding_dimension: int, dropout_probability: float):
        super(DGCNNSemSeg, self).__init__()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(embedding_dimension)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.k = k

        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(
            nn.Conv1d(192, embedding_dimension, kernel_size=1, bias=False),
            self.bn6,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(
            nn.Conv1d(1216, 512, kernel_size=1, bias=False),
            self.bn7,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            self.bn8,
            nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=dropout_probability)
        self.conv9 = nn.Conv1d(256, num_classes, kernel_size=1, bias=False)

    def forward(self, features: torch.Tensor):
        batch_size, _, num_points = features.size()

        # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = get_graph_feature(features, k=self.k, dim9=True)
        # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv1(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x1 = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = get_graph_feature(x1, k=self.k)
        # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv3(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x2 = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = get_graph_feature(x2, k=self.k)
        # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = self.conv6(x)
        # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = x.max(dim=-1, keepdim=True)[0]

        x = x.repeat(1, 1, num_points)                 # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024 + 64 * 3, num_points)

        # (batch_size, 1024 + 64 * 3, num_points) -> (batch_size, 512, num_points)
        x = self.conv7(x)
        # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.conv8(x)
        x = self.dp1(x)
        # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        x = self.conv9(x)
        # (batch_size, 13, num_points) -> (batch_size, num_points, 13)
        x = x.transpose(2, 1).contiguous()

        return x
