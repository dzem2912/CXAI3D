import torch
import numpy as np

from typing import Dict
from dataclasses import dataclass

# Number of parameters to optimize
NUM_CLASS = 9
NUM_FEATURES = 5

VAIHINGEN_CLASS_DICT = {
    'powerlines': 0,
    'grass': 1,  # low vegetation
    'roads': 2,
    'cars': 3,
    'fence': 4,
    'roof': 5,
    'facade': 6,
    'shrub': 7,
    'tree': 8
}


DALES_CLASS_DICT = {
    'unclassified': 0,
    'ground': 1,
    'vegetation': 2,
    'cars': 3,
    'trucks': 4,
    'powerlines': 5,
    'fences': 6,
    'poles': 7,
    'buildings': 8
}

def get_sem_class_name(class_value: int):
    for key, value in DALES_CLASS_DICT.items():
        if value == class_value:
            return key
    raise ValueError('No class name is associated with this value!')

VAIHINGEN_TO_DALES_MAPPING = {
    0: 5,
    1: 1,
    2: 1,
    3: 3,
    4: 6,
    5: 8,
    6: 8,
    7: 2,
    8: 2
}


@dataclass(frozen=True)
class PerturbationData:
    """
    Class which holds the original point cloud, mapping of the points in the
    s segment and sp segment, along with the computed feature vectors
    that have been used in the point_wise_perturbation function.
    The purpose of this class is to save the required data computed on the
    server in PKL format and to be able to unpack on the local device.
    """
    pcd: np.ndarray
    s_sp_mapping: Dict[int, int]
    feature_vectors: np.ndarray

    @classmethod
    def create(cls, pcd: np.ndarray, s_sp_mapping: Dict[int, int], feature_vectors: np.ndarray):
        return cls(pcd, s_sp_mapping, feature_vectors)


@dataclass(frozen=True)
class PerturbationIntensity:
    x: np.float64  # intensity for change on x coordinates
    y: np.float64  # intensity for change on y coordinates
    z: np.float64  # intensity for change on z coordinates
    r: np.float64  # intensity for change on R color channels
    g: np.float64  # intensity for change on G color channels
    b: np.float64  # intensity for change on B color channels
    intensity: np.float64  # intensity for change lidar intensity
    num_returns: np.float64  # intensity for change on number of returns

    @classmethod
    def from_list(cls, values):
        return cls(x=values[0],
                   y=values[1],
                   z=values[2],
                   r=values[3],
                   g=values[4],
                   b=values[5],
                   intensity=values[6],
                   num_returns=values[7])

    @classmethod
    def from_dales_list(cls, values):
        return cls(x=values[0],
                   y=values[1],
                   z=values[2],
                   r=np.float64(0.0),
                   g=np.float64(0.0),
                   b=np.float64(0.0),
                   intensity=values[3],
                   num_returns=np.float64(0.0))


class PointTransformerInput:
    def __init__(self, features: torch.Tensor, device: torch.device):
        if len(features.size()) < 3:
            num_points, in_channels = features.size()
            batch_size = 1
            xyz = features[:, :3]
            feat = features[:, :]

            xyz_flattened = xyz.reshape(num_points, 3).to(device)
            feat_flattened = feat.reshape(num_points, in_channels).to(device)
        else:
            batch_size, in_channels, num_points = features.size()
            xyz = features[:, :3, :]
            feat = features[:, :, :]
            xyz_flattened = xyz.permute(0, 2, 1).reshape(-1, 3).to(device)
            feat_flattened = feat.permute(0, 2, 1).reshape(-1, in_channels).to(device)

        row_splits = torch.tensor([i * num_points for i in range(batch_size + 1)], dtype=torch.int64, device=device)

        self.point = xyz_flattened.contiguous()
        self.feat = feat_flattened.contiguous()
        self.row_splits = row_splits


class SegmentExtractionData:
    def __init__(self, pointcloud_path: str,
                 s_index: int, sp_index: int,
                 box_size_x: float, box_size_y: float,
                 x1: float, y1: float, x2: float, y2: float):
        self.pointcloud_path = pointcloud_path
        self.s_index = s_index
        self.sp_index = sp_index
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2