import os
import sys
import torch
import pickle
import numpy as np
import random
import warnings
import gc

from typing import Dict

from jmetal.util.solution import get_non_dominated_solutions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from source.model_classes import NUM_CLASS, NUM_FEATURES, SegmentExtractionData, PerturbationData
from source.perturbations.pipelines.perturbation_pipeline import PerturbationPipeline

########################################################################################################################
### DALES USE CASES METADATA ###########################################################################################
########################################################################################################################
dales_masking_tuples = [[0.7, 0.8, 0.2, 0.3], [0.0, 0.25, 0.5, 0.75], [0.0, 0.25, 0.5, 0.75]]
dales_s_sp_tuples = [[27644, 73957], [286260, 594710], [594710, 286260]]
dales_box_size_tuples = [[0.3, 0.3], [0.2, 0.2], [0.2, 0.2]]
########################################################################################################################
### VAIHINGEN TUPLES ###################################################################################################
########################################################################################################################
vaihingen_s_sp_tuples = [[35304, 27536], [24503, 7249], [11945, 10001]]
vaihingen_masking_tuples = [[0.25, 0.5, 0.75, 1.0], [0.75, 1.0, 0.5, 0.75], [0.0, 0.25, 0.75, 1.0]]
vaihingen_box_size_tuples = [[0.5, 0.5], [0.3, 0.3], [0.3, 0.3]]
########################################################################################################################
def get_pipeline(segment_extraction_data: SegmentExtractionData, dataset_name: str, algorithm_name: str):
    dales_ckpt_path: str = 'source/semantic_segmentation/models/logs/PointTransformer_dales/training_logs/version_6/checkpoints/epoch=35-step=29988.ckpt'
    vaihingen_ckpt_path: str = 'source/semantic_segmentation/models/logs/PointTransformer_vaihingen/training_logs/version_6/checkpoints/epoch=4-step=55.ckpt'

    model_ckpt_path: str = dales_ckpt_path if dataset_name == 'dales' else vaihingen_ckpt_path
    neural_gas_optimization: bool = True

    return PerturbationPipeline(segment_extraction_data=segment_extraction_data,
                                num_features=NUM_FEATURES,
                                num_classes=NUM_CLASS,
                                model_ckpt_path=model_ckpt_path,
                                neural_gas_optimization=neural_gas_optimization,
                                algorithm_name=algorithm_name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    seeds = [42, 29, 12, 98, 30, 1, 4, 17, 24, 6]
    algorithms = ['nsga3', 'ibea', 'nsga2', 'spea2']

    datasets = ['dales', 'vaihingen']

    overwrite: bool = False

    print("Executing algorithm for the following configuration:")
    for dataset_name in datasets:
        num_samples: int = len(dales_s_sp_tuples) if dataset_name == 'dales' else len(vaihingen_s_sp_tuples)

        for index in range(0, num_samples):
            for algorithm in algorithms:
                for seed in seeds:
                    set_seed(seed)
                    print(f"Algorithm: {algorithm.upper()}")
                    print(f"Dataset: {dataset_name.upper()}")
                    print(f"Seed: {seed}")

                    dales_pcd_path: str = 'perturbation_data/dales/dales_3d_pointcloud.csv'
                    vaihingen_pcd_path: str = 'perturbation_data/vaihingen/remapped/vaihingen_3d_pointcloud_remapped_preprocessed_instances.csv'
                    pcd_path: str = dales_pcd_path if dataset_name == 'dales' else vaihingen_pcd_path

                    s_index = dales_s_sp_tuples[index][0] if dataset_name == 'dales' else vaihingen_s_sp_tuples[index][0]
                    sp_index = dales_s_sp_tuples[index][1] if dataset_name == 'dales' else vaihingen_s_sp_tuples[index][1]

                    print(f"Segment pair: [{s_index}, {sp_index}]")

                    common_root_dir: str = f'source/perturbations/solutions/{dataset_name}/{algorithm}/'

                    perturbation_data_path: str = os.path.join(common_root_dir,
                                                            f'segment_pair_{s_index}_{sp_index}_algo_{algorithm}_run_{seed}_perturbation_data.pkl')
                    solutions_data_path: str = os.path.join(common_root_dir,
                                                            f'segment_pair_{s_index}_{sp_index}_algo_{algorithm}_run_{seed}.pkl')

                    if os.path.exists(perturbation_data_path) and os.path.exists(solutions_data_path) and not overwrite:
                        print(f"This example exists and you have not set it to be overwritten! Skipping ..")
                        continue

                    segment_extraction_data: SegmentExtractionData = SegmentExtractionData(
                        pointcloud_path=pcd_path,
                        s_index=dales_s_sp_tuples[index][0] if dataset_name == 'dales' else vaihingen_s_sp_tuples[index][0],
                        sp_index=dales_s_sp_tuples[index][1] if dataset_name == 'dales' else vaihingen_s_sp_tuples[index][
                            1],
                        box_size_x=dales_box_size_tuples[index][0] if dataset_name == 'dales' else
                        vaihingen_box_size_tuples[index][0],
                        box_size_y=dales_box_size_tuples[index][1] if dataset_name == 'dales' else
                        vaihingen_box_size_tuples[index][1],
                        x1=dales_masking_tuples[index][0] if dataset_name == 'dales' else vaihingen_masking_tuples[index][
                            0],
                        x2=dales_masking_tuples[index][1] if dataset_name == 'dales' else vaihingen_masking_tuples[index][
                            1],
                        y1=dales_masking_tuples[index][2] if dataset_name == 'dales' else vaihingen_masking_tuples[index][
                            2],
                        y2=dales_masking_tuples[index][3] if dataset_name == 'dales' else vaihingen_masking_tuples[index][
                            3])

                    pipeline = get_pipeline(segment_extraction_data, dataset_name, algorithm)

                    problem = pipeline.problem
                    original_pointcloud: np.ndarray = pipeline.pointcloud
                    s_to_sp_mapping: Dict[int, int] = pipeline.s_to_sp_mapping
                    feature_vectors: np.ndarray = problem.feature_vectors

                    perturbation_data: PerturbationData = PerturbationData.create(original_pointcloud, s_to_sp_mapping,
                                                                                feature_vectors)

                    solutions = pipeline.run()

                    solutions = get_non_dominated_solutions(solutions)

                    with open(perturbation_data_path, 'wb') as f:
                        pickle.dump(perturbation_data, f)

                    with open(solutions_data_path, 'wb') as f:
                        pickle.dump(solutions, f)

                    del pipeline, original_pointcloud, solutions, perturbation_data, segment_extraction_data

                    gc.collect()
                    torch.clear_autocast_cache()
                    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
