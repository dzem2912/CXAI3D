import torch
import numpy as np
from typing import Dict

from jmetal.core.solution import FloatSolution
from jmetal.core.problem import FloatProblem

from source.semantic_segmentation.models.semantic_segmentation_model import SemanticSegmentationModel
from source.model_classes import NUM_FEATURES, PointTransformerInput
from source.perturbations.optimization.neural_gas import NeuralGas
from source.perturbations.optimization.optimization_utils import compute_sparsity_jdelser, compute_validity_jdelser, \
    compute_similarity_jdelser

from source.model_classes import PerturbationIntensity
from source.perturbations.perturbation_utils import point_wise_perturbate
from source.utils import sample_neighborhood_indices, softmax


class SegmentPerturbationProblem(FloatProblem):
    def __init__(self,
                 s_to_sp_mapping: dict[int, int],
                 num_parameters: int,
                 pointcloud: np.ndarray,
                 model: SemanticSegmentationModel,
                 neural_gas_optimization: bool):
        super(SegmentPerturbationProblem, self).__init__()

        self.num_objectives = 3
        self.num_constraints = 0
        self.directions = [self.MAXIMIZE] * 3
        self.num_parameters = num_parameters
        self.pointcloud_size: int = 8192
        self.neural_gas_optimization: bool = neural_gas_optimization

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model

        self.s0_segment_indices = np.array([s0_idx for s0_idx in s_to_sp_mapping.keys()])
        self.sp_segment_indices = np.array([s_idx for s_idx in s_to_sp_mapping.values()])
        self.s_to_sp_mapping = s_to_sp_mapping

        self.s0_num_points: int = len(self.s0_segment_indices) + 1

        self.pointcloud = pointcloud

        # Compute feature vectors if you want to optimize with neural gas
        if self.neural_gas_optimization:
            self.num_optimization_points = int(self.s0_num_points * 0.5)
            self.feature_vectors = self.compute_feature_vectors()
        else:
            self.feature_vectors = None
            self.num_optimization_points = self.s0_num_points

        self.num_variables = self.num_parameters * self.num_optimization_points

        # Sample self.pointcloud_size number of points such that the s0_segment_indices + points around the segment
        # make up self.pointcloud_size number of points required for the model input.
        self.pointcloud_indices = sample_neighborhood_indices(self.pointcloud[:, :3], self.s0_segment_indices,
                                                              self.pointcloud_size)
        selected_indices = np.concatenate((self.pointcloud_indices, np.asarray(self.sp_segment_indices)))
        self.pointcloud = self.pointcloud[selected_indices, :]

        # Create a new mapping for segment indices within the reduced point cloud
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
        self.pointcloud_indices = [index_map[idx] for idx in self.pointcloud_indices if idx in index_map]
        self.s0_segment_indices = [index_map[idx] for idx in self.s0_segment_indices if idx in index_map]
        self.sp_segment_indices = [index_map[idx] for idx in self.sp_segment_indices if idx in index_map]

        assert len(self.s0_segment_indices) == len(
            self.sp_segment_indices), "S0 and S segments must be the same length!"

        # Create the updated s_to_sp_mapping with remapped indices
        self.s_to_sp_mapping = {s0_idx: s_idx for s0_idx, s_idx in
                                zip(self.s0_segment_indices, self.sp_segment_indices)}

        self.s0_softmax = self.compute_softmax(self.pointcloud)
        print(f"Pointcloud array shape: {self.pointcloud.shape}")
        print(f"Number of variables: {self.num_variables}")

        self.upper_bound = [1.0] * self.num_variables
        self.lower_bound = [-1.0] * self.num_variables

        # Pre-computing the indices of the closest centroid: they are the same all the time
        self.closest_rep_indices: Dict[int, int] = {}

        if self.feature_vectors is not None:
            for s_point_index in self.s_to_sp_mapping.keys():
                s_feature_vector = self.pointcloud[s_point_index, :8]
                distances = np.linalg.norm(self.feature_vectors - s_feature_vector, axis=1)
                closest_rep_index = np.argmin(distances)
                self.closest_rep_indices[s_point_index] = closest_rep_index
        else:
            self.closest_rep_indices = None

        # Features that are not contained in the current dataset should not be optimized!
        for idx in range(self.num_parameters):
            if np.all(self.pointcloud[:, idx] == 0.0):
                self.upper_bound[idx::self.num_parameters] = [0.0] * (
                        (len(self.upper_bound) - idx) // self.num_parameters + 1)
                self.lower_bound[idx::self.num_parameters] = [0.0] * (
                        (len(self.lower_bound) - idx) // self.num_parameters + 1)

        self.num_active_parameters: int = 0
        for feature_idx in range(3, NUM_FEATURES + 3):
            if not np.all(self.pointcloud[:, feature_idx] == 0):
                self.num_active_parameters += 1

        print(f"Number of active parameters is: {self.num_active_parameters}")

        parameter_names = ['X', 'Y', 'Z', 'R', 'G', 'B', 'Intensity', 'Number of Returns'] * 2
        print(f'Upper and lower bound for first {self.num_parameters * 2} parameters:')
        for idx, parameter_name in enumerate(parameter_names):
            print(f"Bound for {parameter_name}: [{self.lower_bound[idx], self.upper_bound[idx]}]")

    def compute_feature_vectors(self):
        print(f"Computing {self.num_optimization_points} feature vectors ..")
        s0_segment = np.copy(self.pointcloud[self.s0_segment_indices, :8])
        maximum_iterations = len(s0_segment) * 3
        neural_gas = NeuralGas(num_rep_points=self.num_optimization_points, max_iterations=maximum_iterations)
        feature_vectors = neural_gas.fit(s0_segment)
        print(f"Done!")

        return feature_vectors

    def compute_softmax(self, pointcloud: np.ndarray) -> np.ndarray:
        features = torch.Tensor(pointcloud[self.pointcloud_indices, :8].copy()).to(self.device)
        input_features = PointTransformerInput(features, self.device)
        with torch.no_grad():
            output = self.model(input_features).detach().cpu().numpy()
            return softmax(output[self.s0_segment_indices, :])

    def number_of_variables(self) -> int:
        return self.num_variables

    def number_of_objectives(self) -> int:
        return self.num_objectives

    def number_of_constraints(self) -> int:
        return self.num_constraints

    def create_point_parameters(self, index: int, solution: FloatSolution):
        return PerturbationIntensity.from_list(values=solution.variables[index:(index + self.num_parameters)])

    def get_alphas(self, solution: FloatSolution) -> list[PerturbationIntensity]:
        alphas: list[PerturbationIntensity] = []
        for idx in range(self.num_optimization_points):
            param_index = idx * self.num_parameters
            alphas.append(self.create_point_parameters(param_index, solution))

        return alphas

    def point_optimization(self, solution: FloatSolution) -> (float, float, float):
        alphas: list[PerturbationIntensity] = self.get_alphas(solution)

        perturbated_pointcloud = point_wise_perturbate(self.pointcloud[:, :8], self.s_to_sp_mapping,
                                                       self.closest_rep_indices, alphas)

        similarity = self.compute_similarity(perturbated_pointcloud)
        validity = self.compute_validity(perturbated_pointcloud)
        sparsity = self.compute_sparsity(perturbated_pointcloud)

        return similarity, validity, sparsity

    def compute_similarity(self, perturbated_pointcloud: np.ndarray) -> float:
        points_s: np.ndarray = perturbated_pointcloud[self.s0_segment_indices, :3]
        points_s0: np.ndarray = self.pointcloud[self.s0_segment_indices, :3]
        return compute_similarity_jdelser(points_s, points_s0)

    def compute_sparsity(self, perturbated_pointcloud: np.ndarray) -> float:
        s_segment_features: np.ndarray = perturbated_pointcloud[self.s0_segment_indices, 3:8]
        s0_segment_features: np.ndarray = self.pointcloud[self.s0_segment_indices, 3:8]

        # If DALES dataset, then compute sparsity like this!
        if self.num_active_parameters == 1:
            return compute_sparsity_jdelser(s_segment_features[:, 3], s0_segment_features[:, 3])
        else:
            return compute_sparsity_jdelser(s_segment_features, s0_segment_features)

    def compute_validity(self, perturbated_pointcloud: np.ndarray) -> float:
        s_softmax: np.ndarray = self.compute_softmax(perturbated_pointcloud)
        return compute_validity_jdelser(s_softmax, self.s0_softmax)

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        similarity, validity, sparsity = self.point_optimization(solution)

        solution.objectives[0] = similarity
        solution.objectives[1] = validity
        solution.objectives[2] = sparsity

        return solution

    def name(self) -> str:
        return str('SegmentPerturbationProblem')

    @property
    def get_pointcloud(self) -> np.ndarray:
        return self.pointcloud

    @property
    def get_s_sp_mapping(self) -> Dict[int, int]:
        return self.s_to_sp_mapping
