from pathlib import Path

import torch
from torch import device

from datasets.load import get_loaders

from models.cnn_prototype import PrototypeModel
from models.cnn_complex import ComplexModel
from models.cnn_deep import DeepModel

if __name__ == "__main__":
    # Model
    model = DeepModel()

    # Dataset folder name
    dataset_name = 'UTKFace_grouped'

    # Dataloaders
    train_dataloader, validation_dataloader, test_dataloader = get_loaders(dataset_name, batch_size=32)

    # Load state dict
    path = Path(__file__).parent / 'states' / 'deep_bs_16_ne_100_lr_0.001_wd_1e-06_g_0.9999' / f'epoch_{50}'
    model.load_state_dict(torch.load(path))

    # Predict
    features, labels = next(iter(test_dataloader))
    features = features.float()
    labels = labels.float()

    # Print prediction
    print(f'y_true: {labels}')
    print(f'y_pred: {torch.round(model(features))}')
