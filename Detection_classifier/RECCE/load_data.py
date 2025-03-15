from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from torch.utils.data import Subset
import random
from sklearn.model_selection import train_test_split



#################################################################################################


class ImageLabelDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset['train'])

    def __getitem__(self, idx):
        # Récupérer l'image qui est déjà un PngImageFile
        image = self.dataset['train'][idx]['image']

            # Convertir RGBA → RVB si nécessaire
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Pas besoin de réouvrir l'image si elle est déjà au format PngImageFile
        if isinstance(image, Image.Image):
            img = image
        else:
            # Si ce n'est pas un PngImageFile, ouvre l'image
            img = Image.open(image)  

        label = self.dataset['train'][idx]['fake']  # Label 0 ou 1
        
        # Appliquer les transformations sur l'image
        if self.transform:
            if image.mode != "RGB":
                image = image.convert("RGB")
            img = self.transform(img)
        return img, label
    
    def afficher(self,idx):
        image = self.dataset['train'][idx]['image']
        
        # Pas besoin de réouvrir l'image si elle est déjà au format PngImageFile
        if isinstance(image, Image.Image):
            img = image
        else:
            # Si ce n'est pas un PngImageFile, ouvre l'image
            img = Image.open(image)  

        label = self.dataset['train'][idx]['fake']  # Label 0 ou 1
        
        # Appliquer les transformations sur l'image
        if self.transform:
            img = self.transform(img)

        # Si l'image est sous forme de tensor avec la forme (C, H, W), permuter pour (H, W, C)
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()  # Permuter et convertir en numpy pour matplotlib

        plt.imshow(img)
        plt.title(f"Label {label}")
        plt.axis('off')  # Ne pas afficher les axes
        plt.show()
    def count_classes(subset):
        """Compte le nombre d'échantillons par classe dans un torch.utils.data.dataset.Subset."""
    
        full_dataset = subset.dataset

        indices = subset.indices
        # Indices des échantillons dans le subset

        # Récupérer les labels des échantillons du subset
        labels = [full_dataset.dataset['train'][i]['fake'] for i in indices]  # Supposons que le label est à l'index 1

        # Compter le nombre d'occurrences par classe
        class_counts = Counter(labels)
    
        return class_counts
    

#######################################################################################################


if __name__ == "__main__":

    dataset = load_dataset("florian-morel22/deepfake-celeba")
    #print(dataset)
    # Transformation des images et normalisation

    mean = [0.5042836 , 0.4235159 , 0.38155344]
    std = [0.30637124 ,0.28822005, 0.28821015]


    transform = transforms.Compose([
        transforms.Resize((299, 299)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std) ])  

    # Dataset personnalisé
    train_dataset = ImageLabelDataset(dataset, transform=transform)

    #######################################################################################
    #reface_subset (1911 images réelles et 1911 deepfake with REFace )

    real_indices = [i for i in range(len(dataset['train'])) if dataset['train'][i]['model'] == 'Real']
    real_indices = random.sample(real_indices, 1911)  # Sélectionner un sous-ensemble de réels
    fake_indices = [i for i in range(len(dataset['train'])) if dataset['train'][i]['model'] == 'REFace']

    # Diviser chaque classe en train / val / test avec train_test_split
    train_real, temp_real = train_test_split(real_indices, test_size=0.2, random_state=42)
    val_real, test_real = train_test_split(temp_real, test_size=0.5, random_state=42)
    train_fake, temp_fake = train_test_split(fake_indices, test_size=0.2, random_state=42)
    val_fake, test_fake = train_test_split(temp_fake, test_size=0.5, random_state=42)

    # Fusionner les indices pour former les datasets finaux
    train_indices = train_real + train_fake
    val_indices = val_real + val_fake
    test_indices = test_real + test_fake

    # Créer les Subsets
    train_reface_subset = Subset(train_dataset, train_indices)
    val_reface_subset = Subset(train_dataset, val_indices)
    test_reface_subset = Subset(train_dataset, test_indices)

    print(f"Taille du dataset d'entraînement (ReFace) : {len(train_reface_subset)}")
    print(f"Taille du dataset de validation (ReFace) : {len(val_reface_subset)}")
    print(f"Taille du dataset de test (ReFace) : {len(test_reface_subset)}")

    # Vérification des distributions
    train_counts = ImageLabelDataset.count_classes(train_reface_subset)
    val_counts = ImageLabelDataset.count_classes(val_reface_subset)
    test_counts = ImageLabelDataset.count_classes(test_reface_subset)

    print("Distribution dans le train (ReFace) :", train_counts)
    print("Distribution dans la validation (ReFace) :", val_counts)
    print("Distribution dans le test (ReFace) :", test_counts)

    torch.save({
        'train': train_reface_subset,
        'val': val_reface_subset,
        'test': test_reface_subset
    }, "reface_dataset.pth")

    print("Datasets sauvegardés avec succès dans 'reface_dataset.pth' !")

    #######################################################################################
    #e4s_subset (2000 images réelles et 2000 deepfake with e4s )
    real_indices = [i for i in range(len(dataset['train'])) if dataset['train'][i]['model'] == 'Real']
    real_indices = real_indices_subset = random.sample(real_indices, 2000)
    fake_indices = [i for i in range(len(dataset['train'])) if dataset['train'][i]['model'] == 'e4s' ]

    train_real, temp_real = train_test_split(real_indices, test_size=0.2, random_state=42)
    val_real, test_real = train_test_split(temp_real, test_size=0.5, random_state=42)
    train_fake, temp_fake = train_test_split(fake_indices, test_size=0.2, random_state=42)
    val_fake, test_fake = train_test_split(temp_fake, test_size=0.5, random_state=42)

    train_indices = train_real + train_fake
    val_indices = val_real + val_fake
    test_indices = test_real + test_fake

    train_e4s_subset = Subset(train_dataset, train_indices)
    val_e4s_subset = Subset(train_dataset, val_indices)
    test_e4s_subset = Subset(train_dataset, test_indices)

    print(f"Taille du dataset d'entraînement : {len(train_e4s_subset)}")
    print(f"Taille du dataset de validation : {len(val_e4s_subset)}")
    print(f"Taille du dataset de test : {len(test_e4s_subset)}")

    train_counts = ImageLabelDataset.count_classes(train_e4s_subset)
    val_counts = ImageLabelDataset.count_classes(val_e4s_subset)
    test_counts = ImageLabelDataset.count_classes(test_e4s_subset)

    print("Distribution dans le train (E4S) :", train_counts)
    print("Distribution dans la validation (E4S) :", val_counts)
    print("Distribution dans le test (E4S) :", test_counts)

    torch.save({
        'train': train_e4s_subset,
        'val': val_e4s_subset,
        'test': test_e4s_subset
    }, "e4s_dataset.pth")

    print("Datasets sauvegardés avec succès dans 'e4s_dataset.pth' !")

