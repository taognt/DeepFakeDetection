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
import torchvision.utils as vutils

class Mesonet():
    def __init__(
            self,
            network: nn.Module = None,
            device: str = "cpu",
            lr: float = 0.001,
            fraction: float = 1.0,
            batch_size: int = 64,
            trainloader: DataLoader = None,
            epochs_per_cycle: int = 4, # Number of epoch per full cycle
            step_lr: bool = True,
            ):
        self.device = device
        self.lr = lr
        self.fraction = fraction
        self.batch_size = batch_size

        if network is None:
            self.network = Meso4(num_classes=2)
            self.network = self.network.to(device)

        self.optimizer =  optim.Adam(self.network.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-8) # Test SGD
        
        # Scheduler: cyclic 
        if step_lr:
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=8, gamma=0.5)
        
        else:
            # Dynamically calculate step_size based on trainloader
            if trainloader:
                iterations_per_epoch = len(trainloader)  # Number of batches in one epoch
                step_size_up = (epochs_per_cycle // 2) * iterations_per_epoch
                step_size_down = step_size_up  # Symmetric cycle
            else:
                step_size_up = 200  # Fallback value if trainloader is not provided
                step_size_down = 200

            # CyclicLR Scheduler
            self.scheduler = lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=self.lr,         # Minimum learning rate
                max_lr=1e-4,             # Maximum learning rate
                # step_size_up=step_size_up,  
                step_size_down=step_size_down,
                mode='triangular2',      # Smoother decay
                cycle_momentum=False     # Set False for Adam optimizer
            )

        self.criterion = nn.CrossEntropyLoss() # Classification problem: maybe prefer Binary Cross entropy (BCE)

    def train(
            self,
            trainloader: DataLoader,
            valloader: DataLoader,
            nb_epochs: int,
            output_path: str,
            patience: int = 20,         # Stop if no improvement for 10 epochs
            min_delta: float = 0.0005  #0.01  # Minimum change to be considered as improvement
    ):
        print(">>> TRAIN")

        # Create output folder if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Log file
        log_file_path = os.path.join(output_path, "training_log.txt")
        with open(log_file_path, 'a') as log_file:
            log_file.write("\n\n========================== Starting Training\n")
            log_file.write(f"Batch Size: {self.batch_size}, Learning Rate: {self.lr}, Epochs: {nb_epochs} Fraction: {self.fraction}\n")

        best_model_weights = self.network.state_dict()
        best_acc = 0.0

        train_losses = []
        val_losses = []

        train_acc = []
        val_acc = []

        # Initialize graph
        plt.figure(figsize=(10,6))
        
        # Early stopping variables
        patience_counter = 0
        best_loss = float('inf')

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

                # Save the first batch of images for inspection
                if epoch < 10 and i < 10:
                    save_path = os.path.join(output_path, "input_images", f"input_batch_epoch{epoch + 1}_train.png")
                    vutils.save_image(images, save_path, normalize=True)

                self.optimizer.zero_grad()
                outputs = self.network(images)
                _, preds = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()    

                if isinstance(self.scheduler, lr_scheduler.CyclicLR):
                    self.scheduler.step() 

                train_loss += loss.data.item()
                train_corrects += torch.sum(preds == labels.data).to(torch.float32)

                if i % 10 == 0:  # Update progress bar every 10 batches
                    pbar.set_postfix({
                        "Train Loss": round(loss.item(), 4),
                        "Train Acc": round((torch.sum(preds == labels.data).item() / images.size(0)), 4),
                        "Learning Rate": round(self.optimizer.param_groups[0]['lr'], 5)
                    })

            t2 = time.time()
            print(f"-- End training phase: {t2-t1} seconds")
            
            t1 = time.time()
            epoch_train_loss = train_loss / len(trainloader.dataset)
            train_losses.append(epoch_train_loss)
            epoch_train_acc = train_corrects / len(trainloader.dataset)
            train_acc.append(epoch_train_acc)

            t2 = time.time()
            # Scheduler step
            if isinstance(self.scheduler, lr_scheduler.StepLR):
                self.scheduler.step()

            with open(log_file_path, 'a') as log_file:
                log_file.write(f"Epoch {epoch + 1}/{nb_epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, LR: {self.optimizer.param_groups[0]['lr']}\n")


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
                val_acc.append(epoch_val_acc)

                # # Update progress bar for validation
                # pbar.set_postfix({
                #     "Val Loss": round(epoch_val_loss, 4),
                #     "Val Acc": round(epoch_val_acc.item(), 4)
                # }, refresh=True)

                print(f"Val loss: {round(epoch_val_loss, 4)} ")
                print(f"Val acc: {round(epoch_val_acc.item(), 4)}")

                # Early stopping check
                if epoch_val_loss < best_loss - min_delta:
                    best_loss = epoch_val_loss
                    best_acc = epoch_val_acc  # Track best accuracy for logging
                    patience_counter = 0  # Reset patience counter if there's an improvement
                    best_model_weights = self.network.state_dict()
                    torch.save(best_model_weights, os.path.join(output_path, "best_model.pth"))
                else:
                    patience_counter += 1 # No improvement, increment counter

                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"Epoch {epoch + 1}/{nb_epochs} - Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}\n")
                    log_file.write(f"Early stopping - Patience: {patience_counter}/{patience}\n")
                    
                # Stop if patience is exceeded
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

            t2 = time.time()
            print(f"-- End validation phase: {t2-t1}")

            
            # Save model every epoch
            torch.save(best_model_weights, os.path.join(output_path, "models", f"best_model_BS{self.batch_size}_lr{self.lr}.pth"))

            # Graph - Losses
            plt.clf()
            plt.plot(range(1, epoch + 2), train_losses, label="Training Loss", color='blue')
            plt.plot(range(1, epoch + 2), val_losses, label="Validation Loss", color='red')
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title(f"Training and Validation Loss, BS: {self.batch_size}, lr: {self.lr}, {int(self.fraction*100)}% of dataset, best acc: {best_acc}")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(output_path, "loss_graphs_2", f"loss_graph_BS{self.batch_size}_lr{self.lr}_epoch{nb_epochs}_frac{int(self.fraction*100)}%.png"))
            plt.pause(0.1)

            # Graph - Accuracy
            plt.clf()
            plt.plot(range(1, epoch + 2), [acc.item() for acc in train_acc], label="Training Accuracy", color='green')
            plt.plot(range(1, epoch + 2), [acc.item() for acc in val_acc], label="Validation Accuracy", color='orange')
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title(f"Training and Validation Accuracy, BS: {self.batch_size}, lr: {self.lr}, {int(self.fraction*100)}% of dataset, best acc: {best_acc:.4f}")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(output_path, "accuracy_graphs_2", f"accuracy_graph_BS{self.batch_size}_lr{self.lr}_epoch{nb_epochs}_frac{int(self.fraction*100)}%.png"))
            plt.pause(0.1)

        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Best Validation Accuracy: {best_acc:.4f}\n")

        print(f"Best Validation Accuracy: {best_acc:.4f}")

        # Load best model weights
        self.network.load_state_dict(best_model_weights)
        torch.save(self.network.state_dict(), os.path.join(output_path, "models", f"best_model_BS{self.batch_size}_lr{self.lr}.pth"))

        

        