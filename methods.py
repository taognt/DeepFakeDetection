import torch.nn as nn
from Mesonet.network.classifier import Meso4
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm 
from matplotlib import pyplot as plt
import time

class Mesonet():
    def __init__(
            self,
            network: nn.Module = None,
            device: str = "cpu"
            ):
        self.device = device

        if network is None:
            self.network = Meso4(num_classes=2)
            self.network = self.network.to(device)

        self.optimizer =  optim.Adam(self.network.parameters(), lr=0.01, betas=(0.9, 0.99), eps=1e-8)
        self.criterion = nn.CrossEntropyLoss() # Classification problem

    def train(
            self,
            trainloader: DataLoader,
            valloader: DataLoader,
            nb_epochs: int,
            output_path: str
    ):
        print(">>> TRAIN")

        # Create output folder if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        best_model_weights = self.network.state_dict()
        best_acc = 0.0

        train_losses = []
        val_losses = []

        # Training loop
        for epoch in range(nb_epochs):
            self.network.train()  # Set the model to training mode
            train_loss = 0.0
            train_corrects = 0.0
            val_loss = 0.0
            val_corrects = 0.0

            # Training phase
            print(f"-- Training phase")
            t1 = time.time()
            pbar = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f"Epoch {epoch + 1}/{nb_epochs}")
            for i, data in pbar:
                t1bis = time.time()
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.network(images)
                _, preds = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.data.item()
                train_corrects += torch.sum(preds == labels.data).to(torch.float32)

                if i % 10 == 0:  # Update progress bar every 10 batches
                    pbar.set_postfix({
                        "Train Loss": round(loss.item(), 4),
                        "Train Acc": round((torch.sum(preds == labels.data).item() / images.size(0)), 4)
                    })

            t2 = time.time()

            print(f"Last data process time: {t2-t1bis} seconds")
            print(f"-- End training phase: {t2-t1} seconds")
            
            t1 = time.time()
            epoch_train_loss = train_loss / len(trainloader.dataset)
            train_losses.append(epoch_train_loss)
            t2 = time.time()
            print(f"Compute epoch train loss: {t2-t1} seconds.")

            # Validation phase
            t1 = time.time()
            print(f"-- Validation phase")
            self.network.eval()
            with torch.no_grad():
                for images, labels in valloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.network(images)
                    _, preds = torch.max(outputs.data, 1)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.data.item()
                    val_corrects += torch.sum(preds == labels.data).to(torch.float32)

                epoch_val_loss = val_loss / len(valloader.dataset)
                val_losses.append(epoch_val_loss)
                epoch_val_acc = val_corrects / len(valloader.dataset)

                # Update progress bar for validation
                pbar.set_postfix({
                    "Val Loss": round(epoch_val_loss, 4),
                    "Val Acc": round(epoch_val_acc.item(), 4)
                }, refresh=True)

                # Save the best model
                if epoch_val_acc > best_acc:
                    best_acc = epoch_val_acc
                    best_model_weights = self.network.state_dict()
            t2 = time.time()
            print(f"-- End validation phase: {t2-t1}")
            
            # Save model every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save(self.network.state_dict(), os.path.join(output_path, f"epoch_{epoch + 1}.pth"))

        print(f"Best Validation Accuracy: {best_acc:.4f}")

        # Load best model weights
        self.network.load_state_dict(best_model_weights)
        torch.save(self.network.state_dict(), os.path.join(output_path, "best_model.pth"))

        # Plot and save the loss graph
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, nb_epochs + 1), train_losses, label="Training Loss")
        plt.plot(range(1, nb_epochs + 1), val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_path, "loss_graph.png"))
        plt.close()

        