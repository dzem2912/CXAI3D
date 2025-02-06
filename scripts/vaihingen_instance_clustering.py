import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

from scripts.solution_unpacking import labels_to_colors

def cluster_pointcloud_with_semantic_labels(pointcloud, eps: float = 0.01, min_samples: int = 10):
    pointcloud['ins_class'] = -1
    current_instance_id = 0

    for sem_class in pointcloud['sem_class'].unique():
        if np.isnan(sem_class):
            continue

        semantic_mask = pointcloud['sem_class'] == sem_class
        semantic_points = pointcloud.loc[semantic_mask, ['x', 'y', 'z']].values

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(semantic_points)
        cluster_labels = clustering.labels_

        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue

            instance_mask = (cluster_labels == cluster_id)
            point_indices = pointcloud.loc[semantic_mask].index[instance_mask]
            pointcloud.loc[point_indices, 'ins_class'] = current_instance_id
            current_instance_id += 1

    return pointcloud

if __name__ == '__main__':
    # Reload the CSV file to reset the state
    file_path = '../perturbation_data/vaihingen/remapped/vaihingen_3d_pointcloud_remapped_preprocessed.csv'
    pointcloud_df = pd.read_csv(file_path)
    clustered_pointcloud_df = cluster_pointcloud_with_semantic_labels(pointcloud_df, eps=0.01, min_samples=10)
    print(f"{clustered_pointcloud_df.head()}")

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(clustered_pointcloud_df[['x', 'y', 'z']].to_numpy())
    pcd.colors = o3d.utility.Vector3dVector(labels_to_colors(clustered_pointcloud_df['ins_class'].to_numpy()))

    print(f"Unique instances: {np.unique(clustered_pointcloud_df['ins_class']).shape[0]}")
    o3d.visualization.draw_geometries([pcd])

    clustered_pointcloud_df.to_csv(file_path.replace('.csv', '_instances.csv'), index=False)

    file_path = '../perturbation_data/vaihingen/remapped/vaihingen_3d_pointcloud_remapped.csv'
    original_pcd = pd.read_csv(file_path)
    original_pcd['ins_class'] = clustered_pointcloud_df['ins_class'].values

    original_pcd.to_csv(file_path.replace('.csv', '_instances.csv'), index=False)

