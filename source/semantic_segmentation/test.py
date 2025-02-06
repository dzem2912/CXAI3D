import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from source.semantic_segmentation.models.semantic_segmentation_model import SemanticSegmentationModel
from source.utils import get_dataloader


def main(dataset_root: str, checkpoint_path: str, logger_path: str, num_points: int, batch_size: int):
    test_dataloader = get_dataloader(root=dataset_root, num_points=num_points, batch_size=batch_size, train_test_split=False)

    model: SemanticSegmentationModel = SemanticSegmentationModel.load_from_checkpoint(checkpoint_path, strict=False)
    model.batch_size = batch_size
    model.eval()

    logger = CSVLogger(save_dir=logger_path, name='testing_logs')

    trainer = Trainer(logger=logger)
    trainer.test(model=model, dataloaders=test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", type=str, required=True, help="Path to the dataset root directory.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint (.ckpt).")
    parser.add_argument("--logger_path", type=str, required=True, help="Path to save the logging files.")
    parser.add_argument("--num_points", type=int, required=True, help="Number of points the model had for input.")
    parser.add_argument("--batch_size", type=int, required=True, help="Number of batches the model had for training.")

    args = parser.parse_args()

    main(dataset_root=args.dataset_root, checkpoint_path=args.checkpoint_path, logger_path=args.logger_path, num_points=args.num_points, batch_size=args.batch_size)

