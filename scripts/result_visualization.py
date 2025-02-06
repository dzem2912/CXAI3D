import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict

from matplotlib.colors import Normalize
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source.model_classes import PerturbationData, PerturbationIntensity
from source.perturbations.perturbation_utils import point_wise_perturbate


NUM_PARAMETERS = 8

def create_save_csv(pointcloud: np.ndarray, name: str):
    df = pd.DataFrame(pointcloud, columns=['x', 'y', 'z', 'r', 'g', 'b', 'intensity', 'num_returns', 'ins_class', 'sem_class', 'superpixels', 'geom_change', 'feature_change', 'total_change'])
    df.to_csv(name, index=False)

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])


def plot_3d_segment(points: np.ndarray, distances: np.ndarray, use_colorbar: bool = True, file_path: str = ''):
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=distances, cmap='cool', s=10)

    if use_colorbar:
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Intensity')

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    set_axes_equal(ax)
    ax.view_init(elev=0, azim=90)

    ax.grid(False)

    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    if file_path == '':
        plt.show()
    else:
        fig.savefig(file_path)


def compute_intensity_color(original_points, modified_points):
    distances = np.linalg.norm(original_points - modified_points, axis=1)
    norm = Normalize(vmin=distances.min(), vmax=distances.max())
    cmap = plt.get_cmap('viridis')
    print(f"distances shape: {distances.shape}")
    colors = cmap(norm(distances))[:, :3]
    return colors


def normalized_change_metric(original: np.ndarray, perturbed: np.ndarray):
    if original.shape != perturbed.shape:
        raise ValueError("Original and perturbed point clouds must have the same shape.")

    distances = np.linalg.norm(original - perturbed, axis=1)
    scaler = MinMaxScaler()
    d_normalized = scaler.fit_transform(distances.reshape(-1, 1))

    return d_normalized

def get_segments(pointcloud: np.ndarray):
    return np.concatenate((pointcloud[pointcloud[:, -1] == 1], pointcloud[pointcloud[:, -1] == 2]))

def main(args):
    print(f"Preparing data for visualization ..")
    solution_dir: str = f'source/perturbations/solutions/{args.dataset}/{args.algorithm_name}/'
    perturbation_data_name: str = f'segment_pair_{args.s}_{args.sp}_algo_{args.algorithm_name}_run_{args.seed}_perturbation_data.pkl'
    perturbation_solutions_name: str = f'segment_pair_{args.s}_{args.sp}_algo_{args.algorithm_name}_run_{args.seed}.pkl'

    with open(os.path.join(solution_dir, perturbation_data_name), 'rb') as f:
        perturbation_data: PerturbationData = pickle.load(f)

    original_pointcloud: np.ndarray = np.asarray(perturbation_data.pcd)
    s_sp_mapping: Dict[int, int] = perturbation_data.s_sp_mapping
    feature_vectors: np.ndarray = np.asarray(perturbation_data.feature_vectors)

    original_pcd_copy: np.ndarray = np.copy(original_pointcloud)
    original_pcd_copy[:, :3] = MinMaxScaler().fit_transform(original_pcd_copy[:, :3])

    closest_rep_indices = {}
    if feature_vectors is not None:
        for s_index in s_sp_mapping.keys():
            s_point = original_pcd_copy[s_index, :8]
            distances = np.linalg.norm(feature_vectors - s_point, axis=1)
            closest_rep_index = np.argmin(distances)
            closest_rep_indices[s_index] = closest_rep_index
    else:
        closest_rep_indices = None

    del original_pcd_copy

    with open(os.path.join(solution_dir, perturbation_solutions_name), 'rb') as f:
        solutions = pickle.load(f)

    objectives_list = []
    for solution in solutions:
        objectives_list.append(solution.objectives)

    similarity_values = [element[0] for element in objectives_list]
    validity_values = [element[1] for element in objectives_list]
    sparsity_values = [element[2] for element in objectives_list]

    sim_max_idx = similarity_values.index(max(similarity_values))
    val_max_idx = validity_values.index(max(validity_values))
    spa_max_idx = sparsity_values.index(max(sparsity_values))

    similarity_max = solutions[sim_max_idx]
    validity_max = solutions[val_max_idx]
    sparsity_max = solutions[spa_max_idx]

    solutions = [similarity_max, validity_max, sparsity_max]
    solution_names = ['similarity_max', 'validity_max', 'sparsity_max']
    modified_pointclouds = []

    s_segment: np.ndarray = original_pointcloud[original_pointcloud[:, -1] == 1]

    geometric_change_intensity: np.ndarray = normalized_change_metric(s_segment[:, :3],
                                                                      s_segment[:, :3])
    feature_change_intensity: np.ndarray = normalized_change_metric(s_segment[:, 3:8],
                                                                    s_segment[:, 3:8])

    total_change_intensity: np.ndarray = normalized_change_metric(s_segment[:, :8],
                                                                  s_segment[:, :8])

    s_segment = np.hstack((s_segment, geometric_change_intensity.reshape(-1, 1),
                           feature_change_intensity.reshape(-1, 1),
                           total_change_intensity.reshape(-1, 1)))

    print(f"Shape of S segment after: {s_segment.shape}")
    create_save_csv(s_segment, f'segment_pair_{args.s}_{args.sp}_original.csv')
    for idx, solution in enumerate(solutions):
        variables = solution.variables
        print(f"Generating PCD with {solution_names[idx]} parameters ..")

        alphas = []

        for i in range(0, len(variables), NUM_PARAMETERS):
            alphas.append(PerturbationIntensity.from_list(variables[i : i + NUM_PARAMETERS]))

        perturbed_pcd: np.ndarray = np.copy(original_pointcloud)

        scaler = MinMaxScaler()
        perturbed_pcd[:, :3] = scaler.fit_transform(perturbed_pcd[:, :3])
        perturbed_pcd = point_wise_perturbate(perturbed_pcd, s_sp_mapping, closest_rep_indices, alphas)
        perturbed_pcd[:, :3] = scaler.inverse_transform(perturbed_pcd[:, :3])

        perturbed_s_segment: np.ndarray = perturbed_pcd[perturbed_pcd[:, -1] == 1]

        geometric_change_intensity: np.ndarray = normalized_change_metric(s_segment[:, :3],
                                                                          perturbed_s_segment[:, :3])

        feature_change_intensity: np.ndarray = normalized_change_metric(s_segment[:, 6].reshape(-1, 1),
                                                                        perturbed_s_segment[:, 6].reshape(-1, 1))

        total_change_intensity: np.ndarray = normalized_change_metric(
            np.concatenate((s_segment[:, :3], s_segment[:, 6:7]), axis=1),
            np.concatenate((perturbed_s_segment[:, :3], perturbed_s_segment[:, 6:7]), axis=1)
        )

        print(f"Average intensity of geometric change for {solution_names[idx]}: {np.mean(geometric_change_intensity)}")
        print(f"Average intensity of feature change for {solution_names[idx]}: {np.mean(feature_change_intensity)}")
        print(f"Average intensity of geometric and feature change for {solution_names[idx]}: {np.mean(total_change_intensity)}")

        perturbed_s_segment = np.hstack((perturbed_s_segment,
                                         geometric_change_intensity.reshape(-1, 1),
                                         feature_change_intensity.reshape(-1, 1),
                                         total_change_intensity.reshape(-1, 1)))

        modified_pointclouds.append(perturbed_pcd)
        create_save_csv(perturbed_s_segment, f'segment_pair_{args.s}_{args.sp}_{solution_names[idx]}.csv')
        print(f"Done generating PCD with {solution_names[idx]} parameters!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Point Segment Perturbations per Optimization Run')
    parser.add_argument('--dataset', type=str, default='dales',
                        help='Specify from which dataset you want to choose the segments! [CHOICE: vaihingen, dales]')
    parser.add_argument('--algorithm_name', type=str, default='nsga2', help='Specify the algorithm!')
    parser.add_argument('--seed', type=int, default=1, help='Specify the seed!')
    parser.add_argument('--s', type=int, default=27644,
                        help='Choose which value for the S segment you want to choose!')
    parser.add_argument('--sp', type=int, default=73957,
                        help='Choose which value for the Sp segment you want to choose!')
    args = parser.parse_args()
    main(args)