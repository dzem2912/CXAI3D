import numpy as np
from typing import Dict, List

from source.model_classes import PerturbationIntensity


def get_segment_means(pointcloud: np.ndarray, segment_mapping: dict[int, int]):
    s_indices = [x for x in segment_mapping.keys()]
    sp_indices = [x for x in segment_mapping.values()]

    s_points = pointcloud[s_indices, :3]
    sp_points = pointcloud[sp_indices, :3]

    s_centroid = np.mean(s_points, axis=0)
    sp_centroid = np.mean(sp_points, axis=0)

    return s_centroid, sp_centroid


def do_geometric_perturbation(s_centroid: float, sp_centroid: float,
                              s_point: np.ndarray, sp_point: np.ndarray,
                              alpha: PerturbationIntensity):
    s_point -= s_centroid
    sp_point -= sp_centroid

    s_point = s_point + np.array([alpha.x, alpha.y, alpha.z]) * (s_point - sp_point)

    # Add the means back
    s_point = s_point + s_centroid
    sp_point = sp_point + sp_centroid

    return s_point


def do_intensity_perturbation(pointcloud: np.ndarray, s_index: int, sp_index: int, segment_mapping: Dict[int, int],
                              alpha: PerturbationIntensity):
    s_indices = [x for x in segment_mapping.keys()]
    sp_indices = [x for x in segment_mapping.values()]

    s_intensities = pointcloud[s_indices, 6]
    sp_intensities = pointcloud[sp_indices, 6]

    s_intensity_mean = np.mean(s_intensities, axis=0)
    sp_intensity_mean = np.mean(sp_intensities, axis=0)

    s_intensity = pointcloud[s_index, 6] - s_intensity_mean
    sp_intensity = pointcloud[sp_index, 6] - sp_intensity_mean

    s_intensity = s_intensity + alpha.intensity * (s_intensity - sp_intensity)

    s_intensity += s_intensity_mean
    sp_intensity += sp_intensity_mean

    return s_intensity


def do_nr_perturbation(pointcloud: np.ndarray, s_index: int, sp_index: int, segment_mapping: Dict[int, int],
                       alpha: PerturbationIntensity):
    s_indices = [x for x in segment_mapping.keys()]
    sp_indices = [x for x in segment_mapping.values()]

    s_nrs = pointcloud[s_indices, 7]
    sp_nrs = pointcloud[sp_indices, 7]

    s_nr_mean = np.mean(s_nrs, axis=0)
    sp_nr_mean = np.mean(sp_nrs, axis=0)

    s_nr = pointcloud[s_index, 7] - s_nr_mean
    sp_nr = pointcloud[sp_index, 7] - sp_nr_mean

    s_nr = s_nr + alpha.num_returns * (s_nr - sp_nr)

    s_nr += s_nr_mean
    sp_nr += sp_nr_mean

    return s_nr


def do_color_perturbation(pointcloud: np.ndarray, s_index: int, sp_index: int, segment_mapping: Dict[int, int],
                          alpha: PerturbationIntensity):
    s_indices = [x for x in segment_mapping.keys()]
    sp_indices = [x for x in segment_mapping.values()]

    s_colors = pointcloud[s_indices, 3:6]
    sp_colors = pointcloud[sp_indices, 3:6]

    s_color_mean = np.mean(s_colors, axis=0)
    sp_color_mean = np.mean(sp_colors, axis=0)

    s_color = pointcloud[s_index, 3:6] - s_color_mean
    sp_color = pointcloud[sp_index, 3:6] - sp_color_mean

    s_color = s_color + np.array([alpha.r, alpha.g, alpha.b]) * (s_color - sp_color)

    s_color += s_color_mean
    sp_color += sp_color_mean

    return s_color


def point_wise_perturbate(pointcloud: np.ndarray, segment_mapping: Dict[int, int],
                          closest_rep_indices: Dict[int, int],
                          alphas: List[PerturbationIntensity]):
    counter: int = 0
    pointcloud_copy = np.copy(pointcloud)

    for s_point_index, sp_point_index in segment_mapping.items():
        s_coordinate = pointcloud_copy[s_point_index, :3].copy()
        sp_coordinate = pointcloud_copy[sp_point_index, :3].copy()

        if closest_rep_indices is not None:
            closest_rep_index = closest_rep_indices[s_point_index]
            alpha = alphas[closest_rep_index]
        else:
            alpha = alphas[counter]
            counter += 1

        min_z_value = np.min(pointcloud_copy[list(segment_mapping.keys()), 2])
        s_centroid, sp_centroid = get_segment_means(pointcloud_copy, segment_mapping)

        pointcloud_copy[s_point_index, :3] = do_geometric_perturbation(s_centroid, sp_centroid,
                                                                       s_coordinate, sp_coordinate, alpha)

        pointcloud_copy[s_point_index, 2] = np.clip(pointcloud_copy[s_point_index, 2], min_z_value, None)

        if not np.all(pointcloud_copy[:, 3:6] == 0.0):
            pointcloud_copy[s_point_index, 3:6] = do_color_perturbation(pointcloud=pointcloud_copy,
                                                                        s_index=s_point_index,
                                                                        sp_index=sp_point_index,
                                                                        segment_mapping=segment_mapping,
                                                                        alpha=alpha)

        if not np.all(pointcloud_copy[:, 6] == 0.0):
            pointcloud_copy[s_point_index, 6] = do_intensity_perturbation(pointcloud=pointcloud_copy,
                                                                          s_index=s_point_index,
                                                                          sp_index=sp_point_index,
                                                                          segment_mapping=segment_mapping,
                                                                          alpha=alpha)


        if not np.all(pointcloud_copy[:, 7] == 0.0):
            pointcloud_copy[s_point_index, 7] = do_nr_perturbation(pointcloud=pointcloud_copy,
                                                                   s_index=s_point_index,
                                                                   sp_index=sp_point_index,
                                                                   segment_mapping=segment_mapping,
                                                                   alpha=alpha)

    return pointcloud_copy
