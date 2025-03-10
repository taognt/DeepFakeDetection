from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt
from collections import Counter


class ImageLabelDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset['train'])

    def __getitem__(self, idx):
        # Récupérer l'image qui est déjà un PngImageFile
        image = self.dataset['train'][idx]['images']
        
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

        return img, label
    
    def afficher(self,idx):
        image = self.dataset['train'][idx]['images']
        
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
    
        # Récupérer le dataset d'origine et les indices du subset
        full_dataset = subset.dataset  # Le dataset complet

        indices = subset.indices
    # Indices des échantillons dans le subset

        # Récupérer les labels des échantillons du subset
        labels = [full_dataset.dataset['train'][i]['fake'] for i in indices]  # Supposons que le label est à l'index 1

        # Compter le nombre d'occurrences par classe
        class_counts = Counter(labels)
    
        return class_counts



if __name__ == "__main__":

    dataset = load_dataset('florian-morel22/deepfake-diff-gan-vqvae')
    # Charger un dataset localement depuis un chemin spécifique
    # Afficher un aperçu
    print(dataset)

    mean=[0.5246, 0.4264, 0.3750]
    std=[0.2590, 0.2357, 0.2308]


    # Transformation des images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Redimensionner les images à une taille uniforme
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std) ])  # Convertir les images en tenseur

    # Créer le Dataset personnalisé
    train_dataset = ImageLabelDataset(dataset, transform=transform)
    # Calculer la taille du dataset pour chaque partition
    train_size = int(0.7 * len(train_dataset))  
    val_size = int(0.15 * len(train_dataset))
    test_size=len(train_dataset) - train_size - val_size
    # Diviser le dataset en train et validation
    train_subset, val_subset, test_subset = random_split(train_dataset, [train_size, val_size, test_size])

    torch.save({
        'train_dataset': train_subset,
        'val_dataset': val_subset,
        'test_dataset': test_subset
    }, "datasets.pth")

    print("Datasets sauvegardés avec succès dans 'datasets.pth' !")


    ### Exploration 

    print('Le nombre d images entrainées est',len(train_subset))
    print('Le nombre d images pour la validation est',len(val_subset))
    print('Le nombre d images pour les test est',len(test_subset))


    
    # Exemple d'utilisation sur les subsets d'entraînement, validation et test
    train_counts = count_classes(train_subset)
    val_counts = count_classes(val_subset)
    test_counts = count_classes(test_subset)

    print("Distribution dans le train :", train_counts)
    print("Distribution dans la validation :", val_counts)
    print("Distribution dans le test :", test_counts)

#Il y a deux fois plus de 0 que de 1 

#Distribution dans le train : Counter({0: 169, 1: 76}) 245
#Distribution dans la validation : Counter({0: 36, 1: 16}) 52
#Distribution dans le test : Counter({0: 29, 1: 25}) 54

