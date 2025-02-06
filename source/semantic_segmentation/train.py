import torch.nn as nn
from sklearn.model_selection import LeaveOneOut
from torch.utils.data import Subset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping

from source.semantic_segmentation.models.semantic_segmentation_model import SemanticSegmentationModel
from source.utils import *


seed: int = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

learning_rate: float = 0.001
num_points: int = 8192
num_epochs: int = 100

def vaihingen_transfer_learning(model_name: str):
    batch_size: int = 4
    dataset_root: str = '../../training_data/vaihingen/remapped/train/'
    dales_model_ckpt_path: str = 'models/logs/PointTransformer_dales/training_logs/version_6/checkpoints/epoch=35-step=29988.ckpt'

    train_dataloader, val_dataloader = get_dataloader(root=dataset_root, num_points=num_points,
                                                      batch_size=batch_size, train_test_split=True,
                                                      val_split=0.1)

    train_class_weights: torch.Tensor = compute_class_weights(train_dataloader, dataset_root)
    print(f'Class weights loaded: {train_class_weights}')

    model: SemanticSegmentationModel = SemanticSegmentationModel.load_from_checkpoint(dales_model_ckpt_path,
                                                                                      model_name=model_name,
                                                                                      batch_size=batch_size,
                                                                                      num_points=num_points,
                                                                                      learning_rate=learning_rate,
                                                                                      num_epochs=num_epochs,
                                                                                      strict=True)
    model.loss_fn = nn.CrossEntropyLoss(weight=train_class_weights)
    model.lr = learning_rate

    model = model.train()

    logger = CSVLogger(save_dir=f'models/logs/{model_name}_vaihingen/', name='training_logs')
    callbacks = [EarlyStopping(monitor='val_balanced_accuracy', min_delta=1e-3, patience=2, verbose=True, mode='max')]

    trainer = Trainer(max_epochs=num_epochs, logger=logger, callbacks=callbacks)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

def main(model_name: str, dataset_name: str):
    batch_size: int = 4
    dataset_root: str = '../../training_data/vaihingen/remapped/train/'
    model_ckpt_path: str = f'models/logs/{model_name}_dales/training_logs/version_1/checkpoints/epoch=99-step=17500.ckpt'

    train_dataloader, validation_dataloader = get_dataloader(root=dataset_root, num_points=num_points,
                                                             batch_size=batch_size, train_test_split=True, val_split=0.05)

    train_class_weights: torch.Tensor = compute_class_weights(train_dataloader, dataset_root)
    print(f"Class weights loaded: {train_class_weights}")

    model: SemanticSegmentationModel = SemanticSegmentationModel.load_from_checkpoint(model_ckpt_path,
                                                                                      model_name=model_name,
                                                                                      batch_size=batch_size,
                                                                                      num_points=num_points,
                                                                                      learning_rate=learning_rate,
                                                                                      num_epochs=num_epochs,
                                                                                      strict=False)
    model.loss_fn = nn.CrossEntropyLoss(weight=train_class_weights)
    model.lr = learning_rate

    logger = CSVLogger(save_dir=f'models/logs/{model_name}_{dataset_name}/', name='training_logs')
    callbacks = [EarlyStopping(monitor='val_balanced_accuracy', min_delta=1e-3, patience=2, verbose=True, mode='max')]

    trainer = Trainer(max_epochs=num_epochs, logger=logger, callbacks=callbacks)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

if __name__ == '__main__':
    main('PointNet2SemSegMsg', 'vaihingen')
    vaihingen_transfer_learning('PointTransformer')