import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader,Subset
import torch.serialization
# Autoriser explicitement `Subset`
torch.serialization.add_safe_globals([Subset])
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
from torchvision import datasets, models, transforms
from tqdm import tqdm
#
from Mesonet import MesoInception4
from load_data import ImageLabelDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size=32
# Charger les DataLoaders
file_path = os.path.join(os.path.dirname(__file__), "datasets.pth")
data = torch.load(file_path,weights_only=False)
test_subset = data['test_dataset']
test_dataset_size = len(test_subset)
test_loader=DataLoader(test_subset,batch_size=32, shuffle=False)
#
model_path="./models/best_model_inception.pth"

corrects = 0
acc = 0
true_positive=0
false_negative=0
false_positive=0
#
model = MesoInception4()
model.load_state_dict(torch.load(model_path))

model= model.to(device)
model.eval()
with torch.no_grad():
	for (image, labels) in tqdm(test_loader):
		image = image.to(device)
		labels = labels.to(device)
		outputs = model(image)
		_, preds = torch.max(outputs.data, 1)
		corrects += torch.sum(preds == labels.data).to(torch.float32)
		#fake=1
		true_positive += torch.sum((preds == 1) & (labels == 1)).to(torch.float32)
		print('True positive',true_positive)
		# prediction is a real images but it is a fake
		false_negative += torch.sum((preds == 0) & (labels == 1)).to(torch.float32)
		print('False Negative',false_negative)
		# prediction is a fake images but it is a real
		false_positive += torch.sum((preds == 1) & (labels == 0)).item()
		print(false_positive)
	
		batch_size = image.size(0)
		print('Iteration Acc {:.4f}'.format(torch.sum(preds == labels.data).to(torch.float32)/batch_size))
		
	acc = corrects / test_dataset_size
	recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
	precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
	print('Test Acc: {:.4f}'.format(acc))
	print('Test Recall: {:.4f}'.format(recall))
	print('Test Precision: {:.4f}'.format(precision))



