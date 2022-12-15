from pathlib import Path

import torch
from datasets.load import get_loaders

from models.cnn_prototype import PrototypeModel
from models.cnn_complex import ComplexModel
from models.cnn_deep import DeepModel

if __name__ == "__main__":
    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyper parameters
    batch_size = 16

    learning_rate = 1e-3
    weight_decay = 1e-6
    gamma = 0.9999

    n_epochs = 100

    # Model
    model = DeepModel().to(device)

    # Dataset folder name
    dataset_name = 'UTKFace_grouped'

    # Dataloaders
    train_dataloader, validation_dataloader, test_dataloader = get_loaders(dataset_name, batch_size=batch_size)

    # Get state dict
    save_state_dir = f'deep_bs_{batch_size}_ne_{n_epochs}_lr_{learning_rate}_wd_{weight_decay}_g_{gamma}'

    # Load state dict
    path = Path(__file__).parent / 'states' / f'{save_state_dir}' / f'epoch_{50}'
    model.load_state_dict(torch.load(path))

    # Predict
    features, labels = next(iter(test_dataloader))
    features = features.to(device).float()
    labels = labels.to(device).float()

    # Print prediction
    print(f'y_true: {labels}')
    print(f'y_pred: {torch.round(model(features))}')
