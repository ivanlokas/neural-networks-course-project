from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from typing import Tuple, Any


class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Parent returns:
            tuple: (sample, target) where target is class_index of the target class.
        Modified to return:
            tuple: (sample, target) where target is the target class (in order to preserve exact age).
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, int(self.classes[target]) - 1


def get_loaders(dataset_name: str, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Retrieves batch loaders with given batch size

    Args:
        dataset_name (str): Relative dataset path
        batch_size (int): Batch size

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation and test dataloaders
    """

    DATA_DIR = Path(__file__).parent / 'UTKFace'
    DATA_DIR_GRUPED = Path(__file__).parent / dataset_name

    data = CustomImageFolder(DATA_DIR_GRUPED, transform=transforms.ToTensor())

    train_data, validation_data, test_data = random_split(data, [0.5, 0.25, 0.25])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, validation_dataloader, test_dataloader
