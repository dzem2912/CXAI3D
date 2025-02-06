import os
import torch
import numpy as np
from scipy.spatial.ckdtree import cKDTree

from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score,
                             mutual_info_score)

from source.semantic_segmentation.datasets.pointcloud_dataset import PointcloudDataset
from source.model_classes import NUM_CLASS, VAIHINGEN_TO_DALES_MAPPING


def softmax(logits):
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Stability adjustment
    return exps / np.sum(exps, axis=1, keepdims=True)


def map_vaihingen_to_dales(labels):
    """
    Args:
        labels: ISPRS Vaihingen dataset labels that should be mapped to DALES labels
        Returns: ISPRS Vaihingen labels mapped to DALES labels.
    """
    mapped_labels = labels.clone()
    for vaihingen_class, dales_class in VAIHINGEN_TO_DALES_MAPPING.items():
        dales_class_tensor = torch.tensor(dales_class, device=labels.device)
        mapped_labels = torch.where(labels == vaihingen_class, dales_class_tensor, mapped_labels)
    return mapped_labels

def standardize_points(points: np.ndarray) -> np.ndarray:
    """
    Standardizes a set of points to have zero mean and scales them to fit within a unit cube.

    This function centers the points around the origin by subtracting the mean of each dimension
    and then scales the points such that the maximum absolute value of any coordinate is 1.

    Parameters:
    points (np.ndarray): A numpy array of shape (N, 3) representing the coordinates of the points.

    Returns:
    np.ndarray: The standardized points, centered around the origin and scaled to fit within a unit cube.
    """
    centroids = np.mean(points, axis=0)
    points_centered = points - centroids

    max_abs_value = np.max(np.abs(points_centered))
    points = points_centered / max_abs_value

    return points

def compute_class_weights(dataloader: DataLoader, dataset_root: str) -> torch.Tensor:
    train_class_weights_path: str = os.path.join(dataset_root, 'train_class_weights.pth')
    if os.path.exists(train_class_weights_path):
        return torch.load(train_class_weights_path)

    class_counts = Counter()

    for _, labels in tqdm(dataloader):
        flattened_labels = labels.view(-1)
        class_counts.update(flattened_labels.tolist())

    total_samples = sum(class_counts.values())

    class_weights = []
    for class_id in range(NUM_CLASS):
        class_count = class_counts[class_id]
        if class_count > 0:
            weight = total_samples / (NUM_CLASS * class_count)
        else:
            weight = 0.0
        class_weights.append(weight)

    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    if not os.path.exists(train_class_weights_path):
        torch.save(class_weights, train_class_weights_path)

    return class_weights

def get_all_files_with_extension(source_directory_path: str, extension: str = '.csv') -> list[str]:
    file_names: list[str] = []

    for root, dirs, files in os.walk(source_directory_path):
        for file in files:
            if file.endswith(extension.lower()) or file.endswith(extension.upper()):
                file_names.append(os.path.join(root, file))

    return file_names

def dict_to_tensor(class_weights):
    sorted_class_weights = [class_weights[key] for key in sorted(class_weights)]
    return torch.tensor(sorted_class_weights, dtype=torch.float64)


def random_scale_point_cloud(batch_data, scale_low: float = 0.8, scale_high: float = 1.25):
    batch_size, _, _ = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, batch_size)
    for batch_index in range(batch_size):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def shift_point_cloud(batch_data, shift_range: float = 0.1):
    batch_size, _, _ = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (batch_size, 3))
    for batch_index in range(batch_size):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str):
    if metric_name == 'balanced_accuracy_score':
        return balanced_accuracy_score(y_true, y_pred)
    elif metric_name == 'accuracy_score':
        return accuracy_score(y_true, y_pred, average='weighted')
    elif metric_name == 'f1_score':
        return f1_score(y_true, y_pred, average='weighted')
    elif metric_name == 'precision_score':
        return precision_score(y_true, y_pred, average='weighted')
    elif metric_name == 'recall_score':
        return recall_score(y_true, y_pred, average='weighted')
    elif metric_name == 'mutual_info_score':
        return mutual_info_score(y_true, y_pred)
    else:
        raise ValueError(f'Invalid metric name: {metric_name}')

def compute_batch_average_metric(predictions: torch.Tensor, ground_truth: torch.Tensor, metric_name: str):
    batch_size = predictions.shape[0]

    y_true = ground_truth.cpu().data.numpy().flatten()
    y_pred = predictions.cpu().data.numpy().flatten()

    metric_results = []
    if len(predictions.shape) == 3:
        for i in range(batch_size):
            metric_results.append(compute_metric(y_true[i, :], y_pred[i, :], metric_name))
    else:
        metric_results.append(compute_metric(y_true, y_pred, metric_name))

    return np.mean(metric_results)

def compute_single_sample_metric(predictions: torch.Tensor, ground_truth: torch.Tensor, metric_name: str) -> float:
    y_true = ground_truth.cpu().data.numpy().reshape(-1)
    y_pred = predictions.cpu().data.numpy().reshape(-1)

    return compute_metric(y_true, y_pred, metric_name)

def compute_mean_iou(predictions: torch.Tensor, ground_truth: torch.Tensor, num_classes: int):
    """
    Compute the mean IoU for a batch of 3D point semantic segmentation predictions and ground truth.

    :param predictions: Tensor of shape (batch_size, num_points) with predicted class labels.
    :param ground_truth: Tensor of shape (batch_size, num_points) with ground truth class labels.
    :param num_classes: Total number of classes in the segmentation task.
    :return: Mean IoU for the batch.
    """
    batch_size, _ = predictions.shape
    intersections_over_union = []

    for i in range(batch_size):
        iou_per_class = []
        for cls in range(num_classes):
            pred_cls = (predictions[i] == cls)
            gt_cls = (ground_truth[i] == cls)

            intersection = torch.sum(torch.Tensor(pred_cls & gt_cls)).item()
            union = torch.sum(torch.Tensor(pred_cls | gt_cls)).item()

            if union > 0:
                iou = intersection / union + 1e-6
                iou_per_class.append(iou)

        if iou_per_class:
            mean_iou_sample = sum(iou_per_class) / len(iou_per_class)
            intersections_over_union.append(mean_iou_sample)

    mean_iou_batch = sum(intersections_over_union) / len(intersections_over_union) if intersections_over_union else 0
    return mean_iou_batch


def single_sample_mean_iou(predictions: torch.Tensor, ground_truth: torch.Tensor, num_classes: int):
    """
    Compute the mean IoU for a single 3D point semantic segmentation prediction and ground truth.

    :param predictions: Tensor of shape (num_points,) with predicted class labels.
    :param ground_truth: Tensor of shape (num_points,) with ground truth class labels.
    :param num_classes: Total number of classes in the segmentation task.
    :return: Mean IoU for the sample.
    """
    intersections_over_union = []

    for cls in range(num_classes):
        pred_cls = (predictions == cls)
        gt_cls = (ground_truth == cls)

        intersection = torch.sum(torch.Tensor(pred_cls & gt_cls)).item()
        union = torch.sum(torch.Tensor(pred_cls | gt_cls)).item()

        if union > 0:
            iou = intersection / union + 1e-6
            intersections_over_union.append(iou)

    miou = sum(intersections_over_union) / len(intersections_over_union) if intersections_over_union else 0
    return miou

def get_dataloader(root: str, num_points: int, batch_size: int, train_test_split: bool, val_split: float = 0.2):
    dataset: PointcloudDataset = PointcloudDataset(root=root, num_points=num_points)

    if not train_test_split:
       test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)
       print(f"Generated {len(dataset)} testing samples.")
       return test_dataloader

    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)

    print(f"Generated {train_size} training samples and {val_size} validation samples.")

    return train_loader, val_loader

def sample_neighborhood_indices(points: np.ndarray, mandatory_indices: np.ndarray, num_samples: int) -> np.ndarray:
    s0_indices: np.ndarray = np.asarray(mandatory_indices)
    s0_segment_size: int = len(s0_indices)

    if num_samples < s0_segment_size:
        raise ValueError(f'num_samples ({num_samples}) must be greater or equal than s0_segment_size ({s0_segment_size})!')

    remaining_size = num_samples - s0_segment_size
    search_tree = cKDTree(points)

    # Sample defined number of neighbors
    _, neighbors_list = search_tree.query(points[s0_indices], k=num_samples)
    combined_neighbor_indices = np.unique(np.concatenate(neighbors_list[:, 1:]))

    remaining_indices = np.setdiff1d(combined_neighbor_indices, s0_indices)
    remaining_indices_size = len(remaining_indices)

    if remaining_size > remaining_indices_size:
        raise ValueError(
            f'Not enough points to sample the neighborhood! Required: {remaining_size}, available: {remaining_indices_size}')

    down_sampled_indices = np.random.choice(remaining_indices, size=remaining_size, replace=False)
    selected_indices = np.concatenate((s0_indices, down_sampled_indices))

    return selected_indices

