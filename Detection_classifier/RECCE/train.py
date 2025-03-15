import os
import sys
import time
import math
import torch
import random
import numpy as np
from tqdm import tqdm
from pprint import pprint
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from RECCE import Recce
from utils import exp_recons_loss, plot_figure
from utils import  AccMeter, AUCMeter, AverageMeter, Logger,MLLoss
from load_data import ImageLabelDataset



#Chargement dataset
file_path = os.path.join(os.path.dirname(__file__), "reface_dataset.pth")
data = torch.load(file_path, weights_only=False)
train_subset = data['train']
val_subset = data['val']
# Créer des DataLoader pour l'entraînement et la validation
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
print("DataLoaders chargés avec succès !")




# load model
num_classes =1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
###############################################################################
# Hyperparamètres
epochs = 20
batch_size = 32
output_path = './models/'  
model_name = "RECCE_model.pth"
model = Recce(num_classes=num_classes)
model = model.to(device)  



# load optimizer
optimizer = Adam(model.parameters(), lr=0.0002, weight_decay=0.00001)
scheduler = StepLR(optimizer, step_size=22500, gamma=0.5)
loss_criterion=nn.BCEWithLogitsLoss().to(device) 

num_step=1
num_epoch=5
val_step=1

# balance coefficients
lambda_1 = 0.1
lambda_2 = 0.1
#warmup_step = 0


contra_loss = MLLoss()
acc_meter = AccMeter()
loss_meter = AverageMeter()
recons_loss_meter = AverageMeter()
contra_loss_meter = AverageMeter()

best_metric = 0
eval_metric = 'Acc'


def save_ckpt():
    save_dir = os.path.join(output_path, "best_model_reface.pth")
    if not os.path.exists(output_path):
        os.makedirs(output_path)    
    # Sauvegarde le modèle
    torch.save(model.state_dict(), save_dir)

    print('Dataset sauvegardé')


#Train
step=0
for epoch in tqdm(range(epochs)):
    step+=1
    # reset meter
    acc_meter.reset()
    loss_meter.reset()
    recons_loss_meter.reset()
    contra_loss_meter.reset()
    optimizer.step()

    train_generator = enumerate(train_loader, 1)
    print(train_generator)
    # wrap train generator with tqdm for process 0
    #if self.local_rank == 0:
    train_generator = tqdm(train_generator, position=0, leave=True)
    for batch_idx, train_data in train_generator:
        #print(train_data[0])
        global_step = (epoch - 1) * len(train_loader) + batch_idx

        model.train()
        I, Y = train_data
        in_I= I.to(device)
        Y=Y.to(device)
        optimizer.zero_grad()
        Y_pre = model(in_I) 
        # for BCE Setting:
        if num_classes == 1:
            Y_pre = Y_pre.squeeze()
            loss = loss_criterion(Y_pre, Y.float())
            Y_pre = torch.sigmoid(Y_pre)
        else:
            loss = loss_criterion(Y_pre, Y)

        # flood
        loss = (loss - 0.04).abs() + 0.04
        recons_loss = exp_recons_loss(model.loss_inputs['recons'], (in_I, Y))
        contra_loss_value = contra_loss(model.loss_inputs['contra'], Y)  

        loss += lambda_1 * recons_loss + lambda_2 * contra_loss_value
        loss.backward()
        optimizer.step()
        scheduler.step()
        acc_meter.update(Y_pre, Y, num_classes == 1)
        loss_meter.update(loss.item())
        recons_loss_meter.update(recons_loss.item())
        contra_loss_meter.update(contra_loss_value.item())
        iter_acc=acc_meter.mean_acc()

    # log training step
    train_generator.set_description(
        "Train Epoch %d (%d/%d), Global Step %d, Loss %.4f, Recons %.4f, con %.4f, "
        "ACC %.4f, LR %.6f" % (epoch, batch_idx, len(train_loader), global_step,
                            loss_meter.avg, recons_loss_meter.avg,contra_loss_meter.avg,
                            iter_acc , scheduler.get_last_lr()[0]))

    # validating process
    v_idx = random.randint(1, len(val_loader) + 1)
    categories=['original', 'fake']
    model.eval()
    with torch.no_grad():
        acc = AccMeter()
        auc = AUCMeter()
        loss_meter = AverageMeter()
        cur_acc = 0.0  # Higher is better
        cur_auc = 0.0  # Higher is better
        cur_loss = 1e8  # Lower is better
        val_generator = tqdm(enumerate(val_loader, 1), position=0, leave=True)
        for val_idx, val_data in val_generator:
            I, Y = val_data
            in_I= I.to(device)
            Y=Y.to(device)
            Y_pre = model(in_I)
            # for BCE Setting:
            if num_classes == 1:
                Y_pre = Y_pre.squeeze()
                loss = loss_criterion(Y_pre, Y.float())
                Y_pre = torch.sigmoid(Y_pre)
            else:
                loss = loss_criterion(Y_pre, Y)

            acc.update(Y_pre, Y, num_classes == 1)
            auc.update(Y_pre, Y, num_classes == 1)
            loss_meter.update(loss.item())

            cur_acc = acc.mean_acc()
            cur_loss = loss_meter.avg

            val_generator.set_description(
                    "Eval Epoch %d (%d/%d), Global Step %d, Loss %.4f, ACC %.4f" % (
                        epoch, val_idx, len(val_loader), step,
                        cur_loss, cur_acc)
                )

            if val_idx == v_idx or val_idx == 1:

                sample_recons = list()
                for _ in model.loss_inputs['recons']:
                    sample_recons.append(_[:4].to("cpu"))

                # show images
                images = I[:4]
                images = torch.cat([images, *sample_recons], dim=0)
                pred = Y_pre[:4]
                gt = Y[:4]
                figure = plot_figure(images, pred, gt, 4,num_classes, categories, show=True)

        cur_auc = auc.mean_auc()
        print("Eval Epoch %d, Loss %.4f, ACC %.4f, AUC %.4f" % (epoch, cur_loss, cur_acc, cur_auc))
         # record the best acc and the corresponding step
        if eval_metric == 'Acc' and cur_acc >= best_metric:
            best_metric = cur_acc
            best_step = step
            print('Best val Acc: {:.4f}'.format(cur_acc))
            save_ckpt()
        elif eval_metric == 'AUC' and cur_auc >= best_metric:
            best_metric = cur_auc
            best_step = step
            save_ckpt()
        elif eval_metric == 'LogLoss' and cur_loss <= best_metric:
            best_metric = cur_loss
            best_step = step
            save_ckpt()


