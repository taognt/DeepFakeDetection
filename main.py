
import argparse
from pathlib import Path
import os
from methods import Mesonet

from Mesonet.network.transform import mesonet_data_transforms
from utils import load_dataset
import random
from torch.utils.data import Subset
import torch

def main(args):
## -- Parameters -- ##
    data_path = Path(args.data_path)
    fraction = float(args.fraction)
    batch_size = int(args.batch_size)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device used: {device}")

## -- Dataset -- ##
    ## Load dataset ##
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    val_path = os.path.join(data_path, 'val')

    transform_train = mesonet_data_transforms['train']
    transform_val = mesonet_data_transforms['val']

    train_dataset, val_dataset = load_dataset(train_path=train_path, val_path=val_path, transform_train=transform_train, transform_val=transform_val)

    ## Load the subset of the dataset
    train_indices = random.sample(range(len(train_dataset)), int(len(train_dataset)*fraction))
    val_indices = random.sample(range(len(val_dataset)), int(len(val_dataset)*fraction))

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    if fraction < 1.0:
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
        train_dataset_size = len(train_subset)
        val_dataset_size = len(val_subset)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
        train_dataset_size = len(train_dataset)
        val_dataset_size = len(val_dataset)

## -- Model Training -- #
    model = Mesonet(device=device)
    model.train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/valid",
    )

    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to train on."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch Size."
    )

    args = parser.parse_args()

    main(args)