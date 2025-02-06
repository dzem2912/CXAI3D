import torch
import pandas as pd

from source.model_classes import VAIHINGEN_CLASS_DICT
from source.utils import get_all_files_with_extension, map_vaihingen_to_dales


def remap_vaihingen_labels(source: str, destination: str):
    data = pd.read_csv(source)
    labels = torch.Tensor(data['sem_class'].to_numpy())
    remapped_labels: torch.Tensor = map_vaihingen_to_dales(labels)
    data['sem_class'] = remapped_labels.data

    data.to_csv(destination, index=False)

def remap_vaihingen_slices(source, destination):
    train_slices_paths = get_all_files_with_extension(source, '.csv')

    for train_slice_path in train_slices_paths:
        remap_vaihingen_labels(train_slice_path, destination)


def main():
    original_pcd_path: str = '../perturbation_data/vaihingen/vaihingen_3d_pointcloud.txt'
    dataframe = pd.read_csv(original_pcd_path, delimiter=' ', header=None)
    dataframe.columns = ['x', 'y', 'z', 'r', 'g', 'b', 'intensity', 'ins_class', 'num_returns', 'sem_class']
    print(f"Dataframe has {len(dataframe)} rows.")
    dataframe = dataframe[dataframe['sem_class'] != VAIHINGEN_CLASS_DICT['powerlines']]
    print(f"Dataframe has {len(dataframe)} rows.")
    dataframe.to_csv('../perturbation_data/vaihingen/vaihingen_3d_pointcloud.csv', sep=',', index=False)

    remap_vaihingen_labels('../perturbation_data/vaihingen/vaihingen_3d_pointcloud.csv',
                           '../perturbation_data/vaihingen/remapped/vaihingen_3d_pointcloud_remapped.csv')
if __name__ == "__main__":
    main()