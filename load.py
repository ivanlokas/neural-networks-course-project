import torch
import os
from torchvision import datasets, transforms
from typing import Tuple, Any
from group import group_classes

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
        return sample, int(self.classes[target])

def get_loader(path="UTKFace", batch_size=32):
    grouped_dir = path + "_grouped"
    if not os.path.isdir(grouped_dir):
        group_classes(path)
    data = CustomImageFolder(grouped_dir, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(data, batch_size, shuffle=True)
    return loader
