import torch
import torch.nn as nn
import torch.nn.functional as F

from source.semantic_segmentation.models.pointnet2_utils import PointNetFeaturePropagation, PointNetSetAbstractionMsg


# PointNet++ with Multi-Scale Grouping
class PointNet2SemSegMsg(nn.Module):
    def __init__(self, num_classes: int, num_features: int, use_logits: bool = False):
        super(PointNet2SemSegMsg, self).__init__()
        self.use_logits: bool = use_logits

        self.set_abstraction_msg1 = PointNetSetAbstractionMsg(
            num_points=1024,
            radius_list=[0.05, 0.1],
            num_sample_list=[16, 32],
            in_channel=6 + num_features,  # !! NOTE: Adapt based on the task and amount of features you have
            mlp_list=[[16, 16, 32], [32, 32, 64]]
        )

        self.set_abstraction_msg2 = PointNetSetAbstractionMsg(
            num_points=256,
            radius_list=[0.1, 0.2],
            num_sample_list=[16, 32],
            in_channel=32 + 64 + 3,
            mlp_list=[[64, 64, 128], [64, 96, 128]]
        )

        self.set_abstraction_msg3 = PointNetSetAbstractionMsg(
            num_points=64,
            radius_list=[0.2, 0.4],
            num_sample_list=[16, 32],
            in_channel=128 + 128 + 3,
            mlp_list=[[128, 196, 256], [128, 196, 256]]
        )

        self.set_abstraction_msg4 = PointNetSetAbstractionMsg(
            num_points=16,
            radius_list=[0.4, 0.8],
            num_sample_list=[16, 32],
            in_channel=256 + 256 + 3,
            mlp_list=[[256, 256, 512], [256, 384, 512]]
        )

        self.feature_propagation4 = PointNetFeaturePropagation(
            in_channels=512 + 512 + 256 + 256,
            mlp=[256, 256]
        )

        self.feature_propagation3 = PointNetFeaturePropagation(
            in_channels=128 + 128 + 256,
            mlp=[256, 256]
        )

        self.feature_propagation2 = PointNetFeaturePropagation(
            in_channels=32 + 64 + 256,
            mlp=[256, 128]
        )

        self.feature_propagation1 = PointNetFeaturePropagation(
            in_channels=128,
            mlp=[128, 128, 128]
        )

        self.convolution1 = nn.Conv1d(128, 128, 1)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.convolution2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, features: torch.Tensor):
        l0_features = features
        l0_coordinates = features[:, :3, :]  # First 3 columns should be the XYZ coordinates

        # NOTE: l0_features has XYZ + features
        # l0_coordinates has only XYZ, if feature dimensionality is 3, then in_channel is 9
        # XYZ (3 channels) AND XYZ (3 channels) + features (3 channels) = 9
        # Additional 3 features, would result in in_channels = 12
        # print(f"Shape l0_features: {l0_features.size()}; l0_coordinates: {l0_coordinates.size()}")
        l1_coordinates, l1_features = self.set_abstraction_msg1(l0_coordinates, l0_features)
        l2_coordinates, l2_features = self.set_abstraction_msg2(l1_coordinates, l1_features)
        l3_coordinates, l3_features = self.set_abstraction_msg3(l2_coordinates, l2_features)
        l4_coordinates, l4_features = self.set_abstraction_msg4(l3_coordinates, l3_features)

        l3_features = self.feature_propagation4(l3_coordinates, l4_coordinates, l3_features, l4_features)
        l2_features = self.feature_propagation3(l2_coordinates, l3_coordinates, l2_features, l3_features)
        l1_features = self.feature_propagation2(l1_coordinates, l2_coordinates, l1_features, l2_features)
        l0_features = self.feature_propagation1(l0_coordinates, l1_coordinates, None, l1_features)

        x = self.dropout(F.relu(self.batch_norm1(self.convolution1(l0_features))))
        x = self.convolution2(x)
        x = F.log_softmax(x, dim=1) if self.use_logits else F.softmax(x, dim=1)
        x = x.permute(0, 2, 1)

        # If you need the features, return x, l4_features as tuple
        return x

