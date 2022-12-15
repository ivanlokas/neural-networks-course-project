import torch
from torch import nn
from datasets.load import get_loaders
from models.cnn_prototype import PrototypeModel
from util import train

if __name__ == "__main__":
    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyper parameters
    batch_size = 32

    learning_rate = 1e-3
    weight_decay = 1e-3
    gamma = 0.9999

    n_epochs = 5

    # Dataloaders
    train_dataloader, validation_dataloader, test_dataloader = get_loaders(batch_size=batch_size)

    # Model
    model = PrototypeModel().to(device)

    # Model arguments
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)

    # Train model
    train(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=n_epochs
    )
