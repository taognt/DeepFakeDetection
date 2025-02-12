import pandas as pd

from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision.datasets.vision import VisionDataset
from matplotlib import pyplot as plt
import torch
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torchvision

class ImagesDataset(VisionDataset):

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.filenames = dataframe["file"].reset_index(drop=True)
        self.labels = dataframe["label"].reset_index(drop=True)
        self.transform = transform

    def __getitem__(self, index: int):
        img_path = self.filenames[index]
        try:
            X = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

        target = self.labels[index]

        if self.transform:
            X = self.transform(X)

        # Ensure compatibile with mps
        X = X.to(torch.float32) 
        target = torch.tensor(target, dtype=torch.int32)

        return {"pixel_values": X, "labels": target}
    def __showitem__(self, index: int, mean=None, std=None):

        dict_ = self.__getitem__(index)
        image_tensor, label = dict_['pixel_values'], dict_['labels']

        # If normalization parameters are provided, create a denormalization transform
        if mean and std:
            denormalize = v2.Normalize(
                mean=[-m / s for m, s in zip(mean, std)],
                std=[1 / s for s in std]
            )

        # Denormalize if needed
        if mean and std:
            image_tensor = denormalize(image_tensor)

        # Convert the tensor back to a PIL image
        if isinstance(image_tensor, torch.Tensor):
            image = to_pil_image(image_tensor)
        else:
            image = image_tensor

        # Display the image
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()
    
    def __showitems__(self, indices: list, mean=None, std=None):
        plt.figure(figsize=(12, 12))

        # If normalization parameters are provided, create a denormalization transform
        if mean and std:
            denormalize = v2.Normalize(
                mean=[-m / s for m, s in zip(mean, std)],
                std=[1 / s for s in std]
            )

        for i, index in enumerate(indices, start=1):
            dict_ = self.__getitem__(index)
            image_tensor, label = dict_['pixel_values'], dict_['labels']

            # Denormalize if needed
            if mean and std:
                image_tensor = denormalize(image_tensor)

            # Convert the tensor back to a PIL image
            if isinstance(image_tensor, torch.Tensor):
                image = to_pil_image(image_tensor)
            else:
                image = image_tensor

            # Plot the image
            plt.subplot(1, len(indices), i)
            plt.imshow(image)
            plt.title(f"Label: {label}")
            plt.axis("off")
        plt.show()

        
    def __len__(self):
        return len(self.filenames)
    
    def __dataloader__(self, batchsize=10, num_workers=4) -> DataLoader:
        return DataLoader(self, batch_size=batchsize, shuffle=True, num_workers=num_workers)


def load_dataset(train_path: str, val_path: str, transform_train: callable, transform_val: callable):
    """
    This function loads the training and validation datasets from the specified directories,
    applies the respective transformations, and returns the datasets for further use.
    
    Args:
    - train_path (str): Path to the training data directory.
    - val_path (str): Path to the validation data directory.
    - transform_train (callable): Transformation to be applied on each training image.
    - transform_val (callable): Transformation to be applied on each validation image.
    
    Returns:
    - train_dataset (torchvision.datasets.ImageFolder): The training dataset with applied transformations.
    - val_dataset (torchvision.datasets.ImageFolder): The validation dataset with applied transformations.
    """
    train_dataset = torchvision.datasets.ImageFolder(train_path, transform=transform_train)
    val_dataset = torchvision.datasets.ImageFolder(val_path, transform=transform_val)

    return train_dataset, val_dataset