import os
import sys
import pickle
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source.perturbations.perturbation_utils import point_wise_perturbate
from source.model_classes import PerturbationIntensity
from source.utils import get_all_files_with_extension

import random
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_highest_indices(data: np.ndarray):
    similarity_values = [element[0] for element in data]
    validity_values = [element[1] for element in data]
    sparsity_values = [element[2] for element in data]

    max_similarity_index = similarity_values.index(max(similarity_values))
    max_validity_index = validity_values.index(max(validity_values))
    max_sparsity_index = sparsity_values.index(max(sparsity_values))

    return max_similarity_index, max_validity_index, max_sparsity_index

def visualize_multiple_fronts_interactive(objectives_list, destination_dir: str, segment_pair: str):
    traces = []

    for i, objectives in enumerate(objectives_list):
        similarity, validity, sparsity = objectives[:, 0], objectives[:, 1], objectives[:, 2]
        indices = [f'Index: {j}' for j in range(len(objectives))]

        trace = go.Scatter3d(
            x=similarity,
            y=validity,
            z=sparsity,
            mode='markers',
            marker=dict(size=5, opacity=0.8),
            name=f'Run {i + 1}',
            text=indices,
            hoverinfo='text'
        )
        traces.append(trace)

    # Set up the layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='Similarity'),
            yaxis=dict(title='Validity'),
            zaxis=dict(title='Sparsity'),
        ),
        title='Interactive Pareto Fronts for Multiple Runs'
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.write_html(os.path.join(destination_dir, segment_pair + 'all_runs.html'))

def plot_pareto_html(new_data: np.ndarray, destination_dir: str, prefix: str):
    similarity_new = new_data[:, 0]
    validity_new = new_data[:, 1]
    sparsity_new = new_data[:, 2]

    fig = go.Figure(data=[go.Scatter3d(
        x=similarity_new,
        y=validity_new,
        z=sparsity_new,
        mode='markers',
        marker=dict(size=5, color='blue'),
        text=[f"Index: {i}" for i in range(len(similarity_new))],
        hoverinfo='text'
    )])

    fig.update_layout(
        title="Pareto front" + prefix,
        scene=dict(
            xaxis_title="Similarity",
            yaxis_title="Validity",
            zaxis_title="Sparsity"
        ),
        autosize=True
    )

    fig.write_html(os.path.join(destination_dir, prefix + '.html'))

def labels_to_colors(labels: np.ndarray):
    cmap = plt.get_cmap('viridis')
    normalized_labels = labels.astype(float) / labels.max()
    colors = cmap(normalized_labels)
    return MinMaxScaler().fit_transform((colors[:, :3] * 255).astype(np.uint8))

def plot_all_pareto_fronts(solution_dir: str, destination_dir: str, prefix: str):
    all_pickle_files = get_all_files_with_extension(solution_dir, '.pkl')
    print(all_pickle_files)

    solution_sets = [file for file in all_pickle_files if not file.endswith('perturbation_data.pkl')]
    solution_sets = [file for file in solution_sets if prefix in file]
    print(solution_sets)
    all_objectives_list = []
    for solution_set in solution_sets:
        with open(solution_set, 'rb') as f:
            solutions = pickle.load(f)
            objectives_list = []
            for solution in solutions:
                objectives_list.append(np.asarray(solution.objectives))
        all_objectives_list.append(np.asarray(objectives_list))

    visualize_multiple_fronts_interactive(all_objectives_list, destination_dir, prefix)


def visualize_perturbation_results(pcd: np.ndarray):
    points = pcd[:, :3]
    colors = labels_to_colors(pcd[:, -1])

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    pointcloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pointcloud])

def main(algorithm_name: str, dataset_name: str, s_index: int, sp_index: int):
    print(f"Running with the following settings:")
    print(f"  - algorithm_name: {algorithm_name}")
    print(f"  - dataset_name: {dataset_name}")
    print(f"  - s_index: {s_index}")
    print(f"  - sp_index: {sp_index}")
    num_parameters: int = 8
    seeds = [42, 29, 12, 98, 30, 1, 4, 17, 24, 6]

    solution_dir: str = f'source/perturbations/solutions/{dataset_name}/{algorithm_name}/'
    destination_dir: str = os.path.join(solution_dir, 'temp')

    if not os.path.exists(destination_dir):
        print(f"Directory: {destination_dir} does not exist! Creating ..")
        os.makedirs(destination_dir)

    prefix: str = f'segment_pair_{s_index}_{sp_index}_algo_{algorithm_name}_run_'
    plot_all_pareto_fronts(solution_dir, destination_dir, prefix)

    for seed in seeds:
        set_seed(seed)
        current_prefix: str = prefix + f'{seed}'
        perturbation_data_path: str = os.path.join(solution_dir, f'{current_prefix}_perturbation_data.pkl')

        try:
            with open(perturbation_data_path, 'rb') as f:
                perturbation_data = pickle.load(f)
        except Exception as e:
            print(f"Exception occurred! Could not load the following PKL file: {perturbation_data_path}")
            print(f"Exception: {e}")
            continue

        np.save(os.path.join(destination_dir, f'{current_prefix}_original_pointcloud.npy'), perturbation_data.pcd)

        with open(os.path.join(solution_dir, f'{current_prefix}.pkl'), 'rb') as f:
            solutions = pickle.load(f)

        objectives_list = []
        for solution in solutions:
            objectives_list.append(solution.objectives)

        sim_idx, val_idx, spa_idx = find_highest_indices(np.asarray(objectives_list))
        print(f"{sim_idx}, {val_idx}, {spa_idx}")

        print(f"There are {np.asarray(objectives_list).shape} solutions!\n{len(np.unique(np.asarray(objectives_list)))} out of them unique!")
        plot_pareto_html(np.asarray(objectives_list), destination_dir, current_prefix)

        similarity_max = solutions[sim_idx]
        validity_max = solutions[val_idx]
        sparsity_max = solutions[spa_idx]

        solutions = [similarity_max, validity_max, sparsity_max]
        solution_names = ['similarity_max', 'validity_max', 'sparsity_max']

        for idx, solution in enumerate(solutions):
            variables = solution.variables

            alphas = []
            for i in range(0, len(variables), num_parameters):
                alphas.append(PerturbationIntensity.from_list(variables[i : i + num_parameters]))

            solution_pcd = np.copy(perturbation_data.pcd)

            scaler = MinMaxScaler()
            solution_pcd[:, :3] = scaler.fit_transform(perturbation_data.pcd[:, :3])

            closest_rep_indices = {}
            if perturbation_data.feature_vectors is not None:
                for s_point_index in perturbation_data.s_sp_mapping.keys():
                    s_point = solution_pcd[s_point_index, :8]
                    distances = np.linalg.norm(perturbation_data.feature_vectors - s_point, axis=1)
                    closest_rep_index = np.argmin(distances)
                    closest_rep_indices[s_point_index] = closest_rep_index
            else:
                closest_rep_indices = None

            perturbated_pcd = point_wise_perturbate(solution_pcd, perturbation_data.s_sp_mapping,
                                                    closest_rep_indices, alphas)

            perturbated_pcd[:, :3] = scaler.inverse_transform(perturbated_pcd[:, :3])

            print(f"Perturbated PCD shape: {perturbated_pcd.shape}")

            np.save(os.path.join(destination_dir, f'{current_prefix}_perturbated_pcd_{solution_names[idx]}.npy'), perturbated_pcd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm_name", type=str, default="nsga3", help="Name of the optimization algorithm.")
    parser.add_argument("--dataset_name", type=str, default="vaihingen", help="Name of the dataset to use.")
    parser.add_argument("--s_index", type=int, default=11945, help="Start index for the dataset.")
    parser.add_argument("--sp_index", type=int, default=10001, help="Index for a specific dataset point.")

    args = parser.parse_args()

    main(num_parameters=args.num_parameters,
         algorithm_name=args.algorithm_name,
         dataset_name=args.dataset_name,
         s_index=args.s_index,
         sp_index=args.sp_index)




