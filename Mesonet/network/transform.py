from torchvision import transforms
import torch

mesonet_data_transforms = {
    'train': transforms.Compose([
        #transforms.ConvertImageDtype(torch.float32),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        #transforms.ConvertImageDtype(torch.float32),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        #transforms.ConvertImageDtype(torch.float32),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}