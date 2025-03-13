from torch.utils.data import Dataset, DataLoader, Subset
import torch.serialization
# Autoriser explicitement `Subset`
torch.serialization.add_safe_globals([Subset])
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import tqdm
from Mesonet import Meso4
from load_data import ImageLabelDataset

#########################################################################################
# Charger les DataLoaders
# Chemin absolu vers le fichier
file_path = os.path.join(os.path.dirname(__file__), "reface_dataset.pth")
data = torch.load(file_path, weights_only=False)
train_subset = data['train']
val_subset = data['val']

print(len(train_subset))
image, label = train_subset[2003]  # Décompose le tuple en image et label
print(f"Type de l'image : {type(image)}")
print(f"Type du label : {type(label)}")
print(image.shape)
print(label)
# Créer des DataLoader pour l'entraînement et la validation
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
print("DataLoaders chargés avec succès !")


#Train 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############################################################################### Hyperparamètres à modifier
epoches = 20
batch_size = 32
output_path = './models/'  # Spécifier le chemin où enregistrer les modèles
model_name = "mesonet_model_reface.pth"
model = Meso4().to(device)  
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
####################################################################################################

train_dataset_size = len(train_loader.dataset)
val_dataset_size = len(val_loader.dataset)
# Variables pour garder les meilleures performances
best_model_wts = model.state_dict()
best_acc = 0.0
iteration = 0
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
for epoch in tqdm.tqdm(range(epoches)):
	print('Epoch {}/{}'.format(epoch+1, epoches))
	print('-'*10)
	model=model.train()
	train_loss = 0.0
	train_corrects = 0.0
	val_loss = 0.0
	val_corrects = 0.0

	for (image, labels) in train_loader:
		image = image.to(device)
		#print(f"Taille du batch : {image.size(0)}")
		labels = labels.to(device)
		optimizer.zero_grad()
		optimizer.zero_grad()
		#forward pass
		outputs = model(image)
		_, preds = torch.max(outputs, 1)
		# Calcul de la loss
		loss = criterion(outputs, labels)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		 # Mise à jour des métriques
		train_loss += loss.item()
		#train_corrects += torch.sum(preds == labels).to(torch.float32)
		train_corrects += (preds.cpu() == labels.cpu()).sum().item()


	epoch_loss = train_loss / train_dataset_size
	epoch_acc = train_corrects / train_dataset_size

	train_losses.append(epoch_loss)
	train_accuracies.append(epoch_acc)
	print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

	#EVALUATION
	model.eval()
	with torch.no_grad():
		for (image, labels) in val_loader:
			image = image.to(device)
			labels = labels.to(device)
			outputs = model(image)
			_, preds = torch.max(outputs.data, 1)
			loss = criterion(outputs, labels)
			val_loss += loss.item()
			#val_corrects += torch.sum(preds == labels.data).to(torch.float32)
			val_corrects += (preds.cpu() == labels.cpu()).sum().item()

		epoch_loss = val_loss / val_dataset_size
		epoch_acc = val_corrects/ val_dataset_size

		val_losses.append(epoch_loss)
		val_accuracies.append(epoch_acc)
		print('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

	# Sauvegarder le modèle si les performances de validation sont meilleures
	if epoch_acc > best_acc:
		best_acc = epoch_acc
		best_model_wts = model.state_dict()

	# Mise à jour du scheduler
	scheduler.step()

	if epoch % 10 == 0:
		if not os.path.exists(output_path):
			os.makedirs(output_path)
		torch.save(model.state_dict(), os.path.join(output_path, f"{epoch+1}_{model_name}"))
	#Save the model trained with multiple gpu

	# Sauvegarder le meilleur modèle
	print('Best val Acc: {:.4f}'.format(best_acc))
	model.load_state_dict(best_model_wts)
	#torch.save(model.module.state_dict(), os.path.join(output_path, "best.pkl"))
	torch.save(model.state_dict(), os.path.join(output_path, "best_model_reface.pth"))




# Tracer la courbe de loss et d'accuracy
plt.figure(figsize=(12, 5))
# Courbe de perte (loss)
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss", marker="o")
plt.plot(val_losses, label="Validation Loss", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid()
# Courbe d'accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy", marker="o")
plt.plot(val_accuracies, label="Validation Accuracy", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.grid()
# Sauvegarde de l'image au lieu de l'afficher
plt.savefig("training_curves_reface.png")  # Sauvegarde dans un fichier PNG
plt.close()  # Ferme la figure pour éviter d'utiliser trop de mémoire

