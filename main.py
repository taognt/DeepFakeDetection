
import argparse
from pathlib import Path
import os
from methods import Mesonet

from Mesonet.network.transform import mesonet_data_transforms
from utils import load_dataset
import random
from torch.utils.data import Subset
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset as ds_load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torchvision import datasets

class DeepfakeCelebADataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        label = sample["fake"]  # 0 (real), 1 (fake)

        # Convert RGBA to RGB if needed
        if image.mode == "RGBA":
            image = image.convert("RGB")
        elif image.mode == "L":  # If grayscale, convert to RGB
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# Model Testing Function
def test_model(model, test_loader, device):
    model.network.eval()  # Set model to evaluation mode
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model.network(images)
            _, preds = torch.max(outputs, dim=1)  # Get predicted class

            loss = model.criterion(outputs, labels)  # Compute loss
            
            test_loss += loss.item()  # Accumulate loss
            
            all_preds.extend(preds.cpu().numpy())  # Store predictions
            all_labels.extend(labels.cpu().numpy())  # Store ground-truth labels

    # Compute metrics
    avg_test_loss = test_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"\nTest Results: Loss={avg_test_loss:.4f}, Accuracy={accuracy:.4f}, F1 Score={f1:.4f}")

    return avg_test_loss, accuracy, f1

def main(args):
## -- Parameters -- ##
    data_path = Path(args.data_path)
    fraction = float(args.fraction)
    batch_size = int(args.batch_size)
    nb_epochs = int(args.nb_epochs)
    output_path = Path(args.output_path)
    model_path = Path(args.model_path)
    lr = float(args.lr)
    train_model_mode = int(args.train) == 1
    test_model_mode = int(args.test) == 1

    print(f"train: {train_model_mode}")
    print(f"test: {test_model_mode}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    print(f"Device used: {device}")
    print(f"Output name: loss_graph_BS{batch_size}_lr{lr}_frac{int(fraction*100)}%.png")

## -- Dataset -- ##
    ## Load dataset ##
    train_path = os.path.join(data_path, 'Train') #discern Fake / Real
    test_path = os.path.join(data_path, 'Test') #discern Fake / Real
    val_path = os.path.join(data_path, 'Validation') #discern Fake / Real

    transform_train = mesonet_data_transforms['train']
    transform_val = mesonet_data_transforms['val']
    transform_test = mesonet_data_transforms['test']

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
    if train_model_mode:
        model = Mesonet(device=device, fraction=fraction, batch_size=batch_size, lr=lr, trainloader=train_loader)
        model.train(trainloader=train_loader, valloader=val_loader, nb_epochs=nb_epochs, output_path=output_path)
    
    ## Test model on Test dataset from OpenForensic
    test_dataset = datasets.ImageFolder(test_path, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    model = Mesonet(device=device)
    model.network.load_state_dict(torch.load(model_path, map_location=device))

    # Evaluate on test set
    test_loss, test_acc, test_f1 = test_model(model, test_loader, device)

    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}, Test F1-score: {test_f1}")
    
## -- Model Test on generated images from hugging face -- #
    if test_model_mode:
        # Generated images dataset
        ds = ds_load_dataset("florian-morel22/deepfake-celeba") # Used for test
        ds = ds['train']
        real_e4s_images = ds.filter(lambda x: x['model'] != 'REFace')
        real_reface_images = ds.filter(lambda x: x['model'] != 'e4s')
        test_dataset = DeepfakeCelebADataset(ds, transform=transform_test)
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        model = Mesonet(device=device)
        model.network.load_state_dict(torch.load(model_path, map_location=device))

        # Evaluate on test set
        test_loss, test_acc, test_f1 = test_model(model, test_loader, device)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
    )

    parser.add_argument(
        "--train",
        type=int,
        default=True
    )

    parser.add_argument(
        "--test",
        type=int,
        default=False
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate."
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

    parser.add_argument(
        "--model_path",
        type=str,
        default="output_data/best_model.pth"
    )

    parser.add_argument(
        "--nb_epochs",
        type=int,
        default=50,
        help="Number of epochs to train the model on."
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default='output_data',
        help="Output path to store model weights."
    )

    args = parser.parse_args()

    main(args)