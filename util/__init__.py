import os
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


def train(
        model,
        train_dataloader,
        validation_dataloader,
        device,
        criterion,
        optimizer,
        scheduler,
        n_epochs,
        save_state=False,
        save_state_dir=None
) -> None:
    """
    Generic model training function

    Args:
        model: Model that will be trained
        train_dataloader: Training dataloader
        validation_dataloader: Validation dataloader
        device: Device that will be used
        criterion: Criterion function
        optimizer: Optimizer
        scheduler: Scheduler
        n_epochs: Number of epochs
        save_state: True if model should be saved, False otherwise
        save_state_dir: Directory where model will be saved
    """

    start_state_dict = model.state_dict()

    writer = SummaryWriter()

    for epoch in range(n_epochs):
        for index, (features, labels) in enumerate(train_dataloader):
            # Train dataset
            features = features.to(device).float()
            labels = labels.to(device).float()

            # Validation dataset
            validation_features, validation_labels = next(iter(validation_dataloader))
            validation_features = validation_features.to(device).float()
            validation_labels = validation_labels.to(device).float()

            # Forward pass
            outputs = model(features)
            validation_outputs = model.forward(validation_features)

            # Calculate loss
            loss = criterion(outputs, labels)
            validation_loss = criterion(validation_outputs, validation_labels)

            # Update writer
            writer_index = epoch * len(train_dataloader) + index

            writer.add_scalar("Training Loss", loss, writer_index)
            writer.add_scalar("Validation Loss", validation_loss, writer_index)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update elements
            optimizer.step()
            scheduler.step()

        # Save model state
        if save_state:

            # Specify save state directory
            path = Path(__file__).parent.parent / 'states' / save_state_dir

            # Create save state directory, if it does not exist
            if not os.path.exists(path):
                os.makedirs(path)

            # Save starting state
            if epoch == 0:
                torch.save(start_state_dict, path / f'epoch_{epoch}')

            # Save state at end of epoch
            torch.save(model.state_dict(), path / f'epoch_{epoch + 1}')

    writer.close()
