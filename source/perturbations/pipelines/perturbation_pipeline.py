import os
import torch
import numpy as np
import pandas as pd

from typing import Dict
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linear_sum_assignment

# Abstract class implementation
from abc import abstractmethod

from jmetal.core.solution import FloatSolution
from jmetal.operator import UniformMutation, SBXCrossover, PolynomialMutation
from jmetal.util.evaluator import SparkEvaluator
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.termination_criterion import StoppingByEvaluations, StoppingByQualityIndicator

# JMetalPy Relevant imports
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII, UniformReferenceDirectionFactory
from jmetal.algorithm.multiobjective.mocell import MOCell
from jmetal.algorithm.multiobjective.ibea import IBEA
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.neighborhood import C9

from source.perturbations.problem.segment_perturbation_problem import SegmentPerturbationProblem
from source.semantic_segmentation.models.semantic_segmentation_model import SemanticSegmentationModel

from source.model_classes import SegmentExtractionData, DALES_CLASS_DICT, get_sem_class_name


def print_details_for_point_with_index(index, pointcloud):
    """
    # Idx 0 to 2 -> Coordinates
    # Idx 3 to 5 -> Colors
    # Idx 6 -> Intensities
    # Idx 7 -> Number of Returns
    # Idx 8 -> Instance labels
    # Idx 9 -> Semantic labels
    # Idx 10 -> Superpixel labels
    """
    print(f"Point {index}:")
    print(f"Coordinates: {pointcloud[index, 0:3]}")
    print(f"Color: {pointcloud[index, 3:6]}")
    print(f"Intensity: {pointcloud[index, 6]:2f}")
    print(f"Number of returns: {pointcloud[index, 7]:2f}")
    print(f"Instance label: {pointcloud[index, 8]}")
    print(f"Semantic label: {pointcloud[index, 9]}")
    print(f"CLASS: {get_sem_class_name(pointcloud[index, 9])}")
    print("\n")


class PerturbationPipeline:
    def __init__(self, segment_extraction_data: SegmentExtractionData,
                 num_features: int,
                 num_classes: int,
                 model_ckpt_path: str,
                 neural_gas_optimization: bool,
                 algorithm_name: str):

        # Dataclass which contains information on extracting segments from the given PCD
        # s, sp indices, box_size and clipping dimensions of the PCD of interest
        self.segment_extraction_data = segment_extraction_data

        self.neural_gas_optimization = neural_gas_optimization
        self.algorithm_name = algorithm_name

        # Inference model setup
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Optimization relevant parameters
        self.max_evaluations: int = 15000
        self.population_size: int = 100
        self.num_parameters: int = 8

        self.model = SemanticSegmentationModel.load_from_checkpoint(checkpoint_path=model_ckpt_path)
        self.model.eval().cuda()

        self.pointcloud, self.s_to_sp_mapping = self.initialize(self.segment_extraction_data.pointcloud_path)

        # Additional preprocessing step - hinders visualization
        self.problem_pointcloud = self.pointcloud.copy()
        self.problem_pointcloud[:, :3] = MinMaxScaler().fit_transform(self.problem_pointcloud[:, :3])

        self.problem: SegmentPerturbationProblem = self.init_problem()
        self.algorithm = self.init_algorithm()

    def init_algorithm(self):
        if self.algorithm_name.lower() == 'nsga2':
            return NSGAII(
                problem=self.problem,
                population_size=self.population_size,
                offspring_population_size=self.population_size,
                # mutation=UniformMutation(probability=0.15),
                mutation=PolynomialMutation(probability=1.0 / self.num_features),
                crossover=SBXCrossover(probability=0.9),
                termination_criterion=StoppingByEvaluations(max_evaluations=self.max_evaluations))

        if self.algorithm_name.lower() == 'nsga3':
            return NSGAIII(
                problem=self.problem,
                population_size=self.population_size,
                # mutation=UniformMutation(probability=0.15),
                # crossover=SBXCrossover(probability=1.0),
                mutation=PolynomialMutation(probability=1.0 / self.num_features),
                crossover=SBXCrossover(probability=0.9),
                reference_directions=UniformReferenceDirectionFactory(self.problem.number_of_objectives(), n_points=30),
                termination_criterion=StoppingByEvaluations(max_evaluations=self.max_evaluations))

        if self.algorithm_name.lower() == 'ibea':
            return IBEA(
                problem=self.problem,
                kappa=0.05,
                population_size=self.population_size,
                offspring_population_size=self.population_size,
                mutation=PolynomialMutation(probability=1.0 / self.num_features),
                crossover=SBXCrossover(probability=0.9),
                # mutation=UniformMutation(probability=0.15),
                # crossover=SBXCrossover(probability=1.0),
                termination_criterion=StoppingByEvaluations(max_evaluations=self.max_evaluations))

        if self.algorithm_name.lower() == 'spea2':
            return SPEA2(
                problem=self.problem,
                population_size=self.population_size,
                offspring_population_size=self.population_size,
                # mutation=UniformMutation(probability=0.15),
                # crossover=SBXCrossover(probability=1.0),
                mutation=PolynomialMutation(probability=1.0 / self.num_features),
                crossover=SBXCrossover(probability=0.9),
                termination_criterion=StoppingByEvaluations(max_evaluations=self.max_evaluations))

        raise ValueError('Algorithm not supported!')

    @staticmethod
    def get_instance_or_box(pointcloud: np.ndarray, index: int, box_size: tuple[float, float]):
        if pointcloud[index, -1] == DALES_CLASS_DICT['ground']:
            # Get seed coordinates
            points = np.copy(pointcloud[:, :3])
            index_coordinates = points[index]

            original_size = np.max(points, axis=0) - np.min(points, axis=0)
            scale_size = np.min(original_size)
            box_size = (scale_size * box_size[0], scale_size * box_size[1])

            box_size_x_half = box_size[0] / 2
            box_size_y_half = box_size[1] / 2

            box_x_min = index_coordinates[0] - box_size_x_half
            box_x_max = index_coordinates[0] + box_size_x_half
            box_y_min = index_coordinates[1] - box_size_y_half
            box_y_max = index_coordinates[1] + box_size_y_half

            index_mask = ((points[:, 0] >= box_x_min) & (points[:, 0] <= box_x_max) &
                          (points[:, 1] >= box_y_min) & (points[:, 1] <= box_y_max))

            return index_mask

        if not np.all(pointcloud[:, -2] == 0.0):
            return pointcloud[:, -2] == pointcloud[index, -2]

        raise ValueError('You have not implemented selection of points without instance labels :(')

    def initialize(self, file_path: str):
        data = pd.read_csv(file_path, sep=',')

        data = data[data['sem_class'] != DALES_CLASS_DICT['powerlines']]

        assert len(data[data['sem_class'] == DALES_CLASS_DICT['powerlines']]) == 0, 'Powerlines must be excluded!'

        semantic_labels = data['sem_class'].to_numpy().astype(np.int8).reshape(-1, 1)

        if 'ins_class' not in data.columns:
            instance_labels = np.zeros_like(semantic_labels)
        else:
            instance_labels = data['ins_class'].to_numpy().astype(np.int8).reshape(-1, 1)

        points = data[['x', 'y', 'z']].to_numpy()
        colors = data[['r', 'g', 'b']].to_numpy()
        intensities = data['intensity'].to_numpy().reshape(-1, 1)
        num_of_returns = data['num_returns'].to_numpy().reshape(-1, 1)

        assert not np.all(intensities == 0.0), 'Intensities in both datasets must not be zero!'

        normalized_points = MinMaxScaler().fit_transform(points.copy())

        x_min = self.segment_extraction_data.x1
        x_max = self.segment_extraction_data.x2
        y_min = self.segment_extraction_data.y1
        y_max = self.segment_extraction_data.y2

        mask = ((normalized_points[:, 0] >= x_min) & (normalized_points[:, 0] <= x_max) &
                (normalized_points[:, 1] >= y_min) & (normalized_points[:, 1] <= y_max))

        points = points[mask]
        colors = colors[mask]
        intensities = intensities[mask]
        num_of_returns = num_of_returns[mask]
        semantic_labels = semantic_labels[mask]
        instance_labels = instance_labels[mask]

        pointcloud = np.hstack((points, colors, intensities, num_of_returns, instance_labels, semantic_labels))

        file_name: str = file_path.split('/')[-1]
        new_file_name = file_name.replace('.csv', '') + f'_x1_{x_min}_x2_{x_max}_y1_{y_min}_y2_{y_max}.txt'
        file_path = os.path.join(file_path.replace(file_name, ''), new_file_name)

        np.savetxt(file_path, pointcloud)

        s_index = self.segment_extraction_data.s_index
        sp_index = self.segment_extraction_data.sp_index

        print("Point S")
        print_details_for_point_with_index(s_index, pointcloud)

        print("Point S-prime")
        print_details_for_point_with_index(sp_index, pointcloud)

        # TODO: There are edge cases where taking the instances will not work
        #       Case 1: Taking the ground segment as S segment.
        #       Case 2: TBD
        box_size: tuple[float, float] = (
        self.segment_extraction_data.box_size_x, self.segment_extraction_data.box_size_y)

        s_index_mask = self.get_instance_or_box(pointcloud, s_index, box_size)
        sp_index_mask = self.get_instance_or_box(pointcloud, sp_index, box_size)

        s_segment_points = pointcloud[s_index_mask]
        print(f"Shape of s segment: {s_segment_points.shape}")
        sp_segment_points = pointcloud[sp_index_mask]
        print(f"Shape of sp segment: {sp_segment_points.shape}")

        superpixels = np.zeros_like(instance_labels, dtype=np.int8)
        s_superpixel_labels = 1
        sp_superpixel_labels = 2

        superpixels[s_index_mask] = s_superpixel_labels
        superpixels[sp_index_mask] = sp_superpixel_labels

        pointcloud = np.hstack((pointcloud, superpixels))

        s_point_indices = np.where(pointcloud[:, -1] == s_superpixel_labels)[0]
        sp_point_indices = np.where(pointcloud[:, -1] == sp_superpixel_labels)[0]

        s_points = np.copy(pointcloud[:, :3])[pointcloud[:, -1] == s_superpixel_labels]
        sp_points = np.copy(pointcloud[:, :3])[pointcloud[:, -1] == sp_superpixel_labels]

        # Calculate and subtract mean
        s_centroid = np.mean(s_points, axis=0)
        sp_centroid = np.mean(sp_points, axis=0)

        s_points -= s_centroid
        sp_points -= sp_centroid

        print(f"Running the linear assignment algorithm ..")
        distance_matrix = np.linalg.norm(s_points[:, None, :2] - sp_points[None, :, :2], axis=2)

        row_indices, col_indices = linear_sum_assignment(distance_matrix)

        s_to_sp_mapping = {int(s_point_indices[row]): int(sp_point_indices[col]) for row, col in
                           zip(row_indices, col_indices)}

        unique_s_indices_in_mapping = np.unique(list(s_to_sp_mapping.keys()))
        unique_sp_indices_in_mapping = np.unique(list(s_to_sp_mapping.values()))
        print(f"Done!")
        print(f"Size of S: {len(s_points)}, Size of S': {len(sp_points)}")
        print(
            f"{len(unique_s_indices_in_mapping)} unique points in S are matched to {len(unique_sp_indices_in_mapping)} unique points in S-Prime.")

        pointcloud[sp_index_mask, -1] = 0
        pointcloud[list(s_to_sp_mapping.values()), -1] = 2

        file_name: str = file_path.split('/')[-1]
        new_file_name = file_name.replace('.txt', '') + f'_s_index_{s_index}_sp_index_{sp_index}.txt'
        file_path = os.path.join(file_path.replace(file_name, ''), new_file_name)

        np.savetxt(file_path, pointcloud)

        return pointcloud, s_to_sp_mapping

    @abstractmethod
    def init_problem(self):
        return SegmentPerturbationProblem(
            self.s_to_sp_mapping,
            self.num_parameters,
            self.problem_pointcloud,
            self.model,
            self.neural_gas_optimization)

    def run(self):
        algorithm = self.algorithm
        if self.problem is None and self.algorithm is None:
            raise ValueError('Algorithm or Problem is not initialized!')

        progress_bar = ProgressBarObserver(max=self.max_evaluations)
        algorithm.observable.register(observer=progress_bar)
        processes: int = 8
        population_evaluator = SparkEvaluator(processes=processes)
        algorithm.population_evaluator = population_evaluator

        algorithm.run()

        evaluator = algorithm.population_evaluator
        evaluator.spark_context.stop()
        return algorithm.result()
