import torch
from torchvision import transforms
from Mesonet import Meso4
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path="./best_model_reface.pth"
model = Meso4()
#model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.load_state_dict(torch.load(model_path))
# Spécifier le chemin de l'image
image_path='./image_1.jpg'
image = Image.open(image_path)


#fonction mesogip_inference qui prend en entré une image (PIL.Image) et en sortie donne 1 (fake) ou 0 (real)

def mesogip_inference(image,model):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

    #Pre-processing image
	mean = [0.5042836 , 0.4235159 , 0.38155344]
	std = [0.30637124 ,0.28822005, 0.28821015]
	transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std) ])  
	image_tensor=transform(image)
	image_tensor = image_tensor.unsqueeze(0) 
	print(image_tensor.shape) # Ajoute une dimension batch
	image_tensor = image_tensor.to(device)


    #Evaluation
	model.eval()
	outputs = model(image_tensor)
	_, pred = torch.max(outputs.data, 1)
	print(pred)
	if pred.item() == 0:  # Convertit le tenseur en entier
		print("L'image est classée comme une image réelle.")
	elif pred.item() == 1: 
		print("L'image est classée comme une image fausse (deepfake).")
		
	return pred.item()
	

label=mesogip_inference(image,model)
print(label)
	
	
    
	
	
	

