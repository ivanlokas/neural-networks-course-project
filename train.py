import torch
from torch import nn
from datasets.load import get_loaders

from models.cnn_prototype import PrototypeModel
from models.cnn_complex import ComplexModel
from models.cnn_deep import DeepModel

from util import train

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

    # Model arguments
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)

    save_state = True
    save_state_dir = f'deep_bs_{batch_size}_ne_{n_epochs}_lr_{learning_rate}_wd_{weight_decay}_g_{gamma}_{"second"}'

    # Train model
    train(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=n_epochs,
        save_state=save_state,
        save_state_dir=f'{save_state_dir}'
    )
