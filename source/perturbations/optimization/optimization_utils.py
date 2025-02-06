import torch
import numpy as np

# Visualization
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import open3d as o3d

########################################################################################################################
# Final implementation of objective functions ##########################################################################
#########################################################################################################################
def compute_sparsity_jdelser(segment_s: np.ndarray, segment_s0: np.ndarray, epsilon: float = 0.05) -> float:
    assert segment_s.shape == segment_s0.shape, "Segments must have the same shape."

    if len(segment_s.shape) < 2:
        num_points = segment_s.shape[0]
        num_features: int = 1
    else:
        num_points, num_features = segment_s0.shape

    matching_count = np.sum(np.abs(segment_s - segment_s0) <= epsilon)
    sparsity = matching_count / (num_points * num_features)
    return sparsity

def compute_validity_jdelser(softmax_s: np.ndarray, softmax_s0: np.ndarray) -> float:
    assert softmax_s.shape == softmax_s0.shape, "Softmax outputs must have the same shape."

    total_diff_sum = 0.0
    num_points = softmax_s.shape[0]

    for softmax_x, softmax_x0 in zip(softmax_s, softmax_s0):
        c_max_original = np.argmax(softmax_x0)
        total_diff_sum += np.abs(softmax_x[c_max_original] - softmax_x0[c_max_original])

    validity = total_diff_sum / num_points
    return validity


def compute_similarity_jdelser(segment_s: np.ndarray, segment_s0: np.ndarray) -> float:
    assert segment_s.shape == segment_s0.shape, "Both segments must have the same number of points."
    total_similarity: float = 0.0

    for x_p in segment_s0:
        distances = np.linalg.norm(segment_s - x_p, axis=1)
        min_distance = np.min(distances)
        total_similarity += min_distance

    return -1.0 * (total_similarity / segment_s.shape[0])
########################################################################################################################
def visualize_model_output(points: np.ndarray, softmax_predictions: np.ndarray, ground_truth: np.ndarray):
    from sklearn.metrics import balanced_accuracy_score
    print(f"Softmax predictions shape: {softmax_predictions.shape}")
    labels = np.argmax(softmax_predictions, axis=1)
    colormap = plt.cm.get_cmap('rainbow', softmax_predictions.shape[1])
    print(f"The color map: {colormap}")
    colors = colormap(labels / labels.max())[:, :3]
    print(f"Balanced accuracy score: {balanced_accuracy_score(ground_truth, labels) * 100:.2f}")

    label_color_mapping = {}
    for label, color in zip(labels, colors):
        label_color_mapping[label] = color

    print(f"Label to color mapping: {label_color_mapping}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])


def visualize_pareto_front(objectives):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], color='red', label='Pareto Front')

    ax.set_title('Solution sample')
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Validity')
    ax.set_zlabel('Similarity')
    ax.legend()
    plt.show()


def visualize_pareto_front_3d(objectives):
    similarity, validity, sparsity = objectives[:, 0], objectives[:, 1], objectives[:, 2]
    indices = [f'Index: {i}' for i in range(len(objectives))]

    # Create a 3D scatter plot for Pareto front
    trace = go.Scatter3d(
        x=similarity,
        y=validity,
        z=sparsity,
        mode='markers',
        marker=dict(size=5, color='red', opacity=0.8),
        text=indices,
        hoverinfo='text'
    )

    # Set up the layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='Similarity'),
            yaxis=dict(title='Validity'),
            zaxis=dict(title='Sparsity'),
        ),
        title='Interactive Pareto Front'
    )

    fig = go.Figure(data=[trace], layout=layout)
    fig.show()


def get_model_input(features: np.ndarray, device: str) -> torch.Tensor:
    features = torch.tensor(features, dtype=torch.float32)
    batch = torch.stack([features, features], dim=0)

    return batch.transpose(1, 2).to(device)


# FOR TESTING! #########################################################################################################
def generate_softmax_output(random_scores: np.ndarray) -> np.ndarray:
    return np.exp(random_scores) / np.sum(np.exp(random_scores), axis=1, keepdims=True)


def generate_pointcloud(num_points: int) -> np.ndarray:
    xyz = np.random.uniform(-1, 1, (num_points, 3))
    rgb = np.random.uniform(0, 1, (num_points, 3))
    intensity = np.random.uniform(0, 1, (num_points, 1))
    class_labels = np.random.randint(0, 9, (num_points, 1))

    return np.hstack((xyz, rgb, intensity, class_labels))


def main():
    num_points: int = 1000
    num_classes: int = 9
    random_scores_s0 = np.random.rand(num_points, num_classes)
    random_scores_s = np.random.rand(num_points, num_classes)

    softmax_s = generate_softmax_output(random_scores_s)
    softmax_s0 = generate_softmax_output(random_scores_s0)

    print(f"Softmax S: {softmax_s.shape}; Softmax S0: {softmax_s0.shape}")

    print(f"Computing Softmax Validity ..")
    validity = compute_validity_jdelser(softmax_s, softmax_s0)
    print(f"Validity F: {validity}")

    pointcloud1 = generate_pointcloud(num_points)
    pointcloud2 = generate_pointcloud(num_points)

    assert pointcloud1.shape == pointcloud2.shape  # Do they have to?

    print(f"Computing Point Similarity ..")
    similarity = compute_similarity_jdelser(pointcloud1[:, :3], pointcloud2[:, :3])
    print(f"Similarity F: {similarity}")

    feature_sparsity_score = []
    for feature in range(pointcloud1.shape[1] - 1):
        feature_sparsity = compute_sparsity_jdelser(pointcloud1, pointcloud2)
        feature_sparsity_score.append(feature_sparsity)

    print(f"Feature-wise Sparsity F: {feature_sparsity_score}")
    print(f"Sparsity F: {np.sum(feature_sparsity_score)}")


if __name__ == "__main__":
    main()


