import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl

from typing import Optional

# Input formating and defined values
from source.model_classes import PointTransformerInput, NUM_CLASS, NUM_FEATURES

# Model implementations
from source.semantic_segmentation.models.dgcnn_sem_seg import DGCNNSemSeg
from source.semantic_segmentation.models.pointnet2_sem_seg import PointNet2SemSegMsg
from open3d.ml.torch.models import PointTransformer

# Schedulers for training
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, CosineAnnealingWarmRestarts
from source.semantic_segmentation.models.model_utils import inplace_relu, init_pt_weights, init_pointnet_weights, init_dgcnn_weights
# Data augmentation
from source.utils import random_scale_point_cloud, shift_point_cloud

# Model evaluation metrics
from source.utils import compute_batch_average_metric, compute_mean_iou

def get_model(model_name: str, device: torch.device):
    if model_name == 'PointNet2SemSegMsg':
        model = PointNet2SemSegMsg(num_classes=NUM_CLASS,
                                  num_features=NUM_FEATURES).to(device)
        model.apply(init_pointnet_weights)
        model.apply(inplace_relu)
        return model

    if model_name == 'DGCNNSemSeg':
        model = DGCNNSemSeg(num_classes=NUM_CLASS, k=20, embedding_dimension=1024,
                           dropout_probability=0.2).to(device)
        model.apply(init_dgcnn_weights)
        model.apply(inplace_relu)
        return model


    if model_name == 'PointTransformer':
        model = PointTransformer(in_channels=(3 + 3 + NUM_FEATURES), num_classes=NUM_CLASS).to(device)
        model.apply(init_pt_weights)
        model.apply(inplace_relu)
        return model

    raise ValueError(f'Model: {model_name} not supported!')

class SemanticSegmentationModel(pl.LightningModule):
    def __init__(self, model_name: str, batch_size: int, num_points: int,
                 learning_rate: float, num_epochs: int,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        # Save everything regarding the model - Weights + parameters (e.g. learning rate, batch size, etc.)
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.num_points = num_points
        self.lr = learning_rate
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.weight = weight

        self.model = get_model(model_name, self.device)
        self.loss_fn = nn.CrossEntropyLoss(weight=weight).to(self.device)

    def augment_features(self, features: torch.Tensor, train_step: bool) -> torch.Tensor | PointTransformerInput:
        if train_step:
            numpy_features: np.ndarray = features.cpu().data.numpy()
            numpy_features[:, :, 0:3] = random_scale_point_cloud(numpy_features[:, :, 0:3])
            numpy_features[:, :, 0:3] = shift_point_cloud(numpy_features[:, :, 0:3])
            features = torch.Tensor(numpy_features)

        features = torch.Tensor(features).float().to(self.device)
        features = features.transpose(2, 1)

        if self.model_name == 'PointTransformer':
            return PointTransformerInput(features, self.device)

        return features

    def test_step(self, batch: torch.Tensor, batch_idx: torch.Tensor):
        features, targets = batch
        features = self.augment_features(features, train_step=False)
        targets: torch.Tensor = torch.Tensor(targets).long().to(self.device).squeeze(-1)

        seg_probabilities = self(features)

        if len(seg_probabilities.shape) < 3:
            seg_probabilities = seg_probabilities.unsqueeze(0)

        if self.model_name == 'PointTransformer':
            seg_probabilities = seg_probabilities.reshape(self.batch_size, self.num_points, NUM_CLASS)

        predicted_labels: torch.Tensor = torch.argmax(seg_probabilities, dim=2)

        self.log('test_balanced_accuracy', compute_batch_average_metric(predicted_labels, targets,
                                                                       'balanced_accuracy_score'),
                 prog_bar=True,
                 logger=True,
                 on_epoch=True)

        self.log('test_f1_score', compute_batch_average_metric(predicted_labels, targets,
                                                              'f1_score'),
                 logger=True,
                 on_epoch=True)

        self.log('test_precision_score', compute_batch_average_metric(predicted_labels, targets,
                                                                     'precision_score'),
                 logger=True,
                 on_epoch=True)

        self.log('test_recall_score', compute_batch_average_metric(predicted_labels, targets,
                                                                  'recall_score'),
                 logger=True,
                 on_epoch=True)

        self.log('test_mutual_info_score', compute_batch_average_metric(predicted_labels, targets,
                                                                       'mutual_info_score'),
                 logger=True,
                 on_epoch=True)

        self.log('test_mean_iou', compute_mean_iou(predicted_labels, targets, NUM_CLASS),
                 logger=True,
                 on_epoch=True)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        features, targets = batch
        features = self.augment_features(features, train_step=True)
        targets: torch.Tensor = torch.Tensor(targets).long().to(self.device).squeeze(-1)

        seg_probabilities = self(features)
        if len(seg_probabilities.shape) < 3:
            seg_probabilities = seg_probabilities.unsqueeze(0)

        if self.model_name == 'PointTransformer':
            seg_probabilities = seg_probabilities.reshape(self.batch_size, self.num_points, NUM_CLASS)

        predicted_labels: torch.Tensor = torch.argmax(seg_probabilities, dim=2)

        seg_probabilities = seg_probabilities.contiguous().view(-1, NUM_CLASS)

        loss = self.loss_fn(seg_probabilities, targets.flatten())

        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        self.log('train_balanced_accuracy', compute_batch_average_metric(predicted_labels, targets,
                                                                         'balanced_accuracy_score'),
                 prog_bar=True,
                 logger=True,
                 on_epoch=True,
                 on_step=False)

        self.log('train_f1_score', compute_batch_average_metric(predicted_labels, targets,
                                                                'f1_score'),
                 logger=True,
                 on_epoch=True,
                 on_step=False)

        self.log('train_precision_score', compute_batch_average_metric(predicted_labels, targets,
                                                                       'precision_score'),
                 logger=True,
                 on_epoch=True,
                 on_step=False)

        self.log('train_recall_score', compute_batch_average_metric(predicted_labels, targets,
                                                                    'recall_score'),
                 logger=True,
                 on_epoch=True,
                 on_step=False)

        self.log('train_mutual_info_score', compute_batch_average_metric(predicted_labels, targets,
                                                                         'mutual_info_score'),
                 logger=True,
                 on_epoch=True,
                 on_step=False)

        self.log('train_mean_iou', compute_mean_iou(predicted_labels, targets, NUM_CLASS),
                 logger=True,
                 on_epoch=True,
                 on_step=False)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        features, targets = batch
        features = self.augment_features(features, train_step=False)
        targets: torch.Tensor = torch.Tensor(targets).long().to(self.device).squeeze(-1)
        seg_probabilities = self(features)

        if self.model_name == 'PointTransformer':
            if len(seg_probabilities.shape) < 3:
                seg_probabilities = seg_probabilities.unsqueeze(0)
            seg_probabilities = seg_probabilities.reshape(self.batch_size, self.num_points, NUM_CLASS)

        predicted_labels: torch.Tensor = torch.argmax(seg_probabilities, dim=2)

        self.log('val_balanced_accuracy', compute_batch_average_metric(predicted_labels, targets,
                                                                       'balanced_accuracy_score'),
                 prog_bar=True,
                 logger=True,
                 on_epoch=True)

        self.log('val_f1_score', compute_batch_average_metric(predicted_labels, targets,
                                                               'f1_score'),
                 logger=True,
                 on_epoch=True)

        self.log('val_precision_score', compute_batch_average_metric(predicted_labels, targets,
                                                                     'precision_score'),
                 logger=True,
                 on_epoch=True)

        self.log('val_recall_score', compute_batch_average_metric(predicted_labels, targets,
                                                                  'recall_score'),
                 logger=True,
                 on_epoch=True)

        self.log('val_mutual_info_score', compute_batch_average_metric(predicted_labels, targets,
                                                                       'mutual_info_score'),
                 logger=True,
                 on_epoch=True)

        self.log('val_mean_iou', compute_mean_iou(predicted_labels, targets, NUM_CLASS),
                 logger=True,
                 on_epoch=True)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = None
        if self.model_name == 'DGCNNSemSeg':
            scheduler = CosineAnnealingLR(optimizer, self.num_epochs, eta_min=self.lr)
        if self.model_name == 'PointNet2SemSegMsg':
            scheduler = StepLR(optimizer, step_size=20, gamma=0.7)
        if self.model_name == 'PointTransformer':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        if scheduler is None:
            raise ValueError('Model not yet supported!')

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }



