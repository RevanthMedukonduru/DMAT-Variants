import os
import shutil
import logging
from tqdm import tqdm
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
from datasets.mixed10_natural import Dataset as test_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.loaders import \
    set_device, move_to_device, set_random_seed, load_config, load_optimizer, \
    get_classifier, get_dataset, get_transform, get_attack
from utils.metrics import AverageMeter
from stylegan_old.stylegan_generator_model import StyleGANGeneratorModel
from advertorch.attacks import PGDAttack
from advertorch.context import ctx_noparamgrad_and_eval
import torchvision
from torchvision.utils import save_image

import torch.nn as nn
import copy

import dnnlib, legacy
import wandb
from torchvision import transforms

from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# parse command line options
parser = argparse.ArgumentParser(description="Training using ImageNet-10 - SINGLE NET")
parser.add_argument("--config", default="our_experiments/classifiers/single_net/clean/clean_config.yml")
parser.add_argument("--resume", default="runs/classifiers/single_networks/R50/Mix10/clean_models/mixed10_classifier_e_9.pt")
args = parser.parse_args()

cfg = load_config(args.config)

trainset_cfg = cfg.dataset.train
testset_cfg = cfg.dataset.test
print(cfg)

# Setup for Wandb
wandb.init(project=f"{cfg.dataset.name}_CLASSIFIERS_{cfg.network.name}_F", config=cfg)
logging.info(cfg)

output_dir = cfg.classifier.path
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, 'train_log.txt')
vis_dir = os.path.join(output_dir, 'vis')
os.makedirs(vis_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    filename=output_filename,
                    filemode='w' if not args.resume else 'a')

# set device and random seed
set_device(cfg, device)
set_random_seed(cfg)
cudnn.benchmark = True

# set classifier
net = get_classifier(cfg, cfg.classifier)
net = net.to(device)

# set loss
criterion = torch.nn.CrossEntropyLoss().to(device)

# set optimizers
optimizer = load_optimizer(cfg.optimizer, params=net.parameters())

# LR scheduler
# lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)
 
start_epoch = 0
best_train_acc_c, best_train_loss_c = 0.0, 0.0
best_test_acc_c, best_test_loss_c = 0.0, 0.0
best_train_epoch_c = 0.0 
best_test_epoch_c = 0.0
if args.resume:
    print("=> loading checkpoint resuming '{}'".format(args.resume))
    ckpt = torch.load(args.resume)
    net.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch']+1
    best_train_acc_c = ckpt['best_train_acc_c']
    best_train_loss_c = ckpt['best_train_loss_c']
    best_test_acc_c = ckpt['best_test_acc_c']
    best_test_loss_c = ckpt['best_test_loss_c']
    best_train_epoch_c = ckpt['best_train_epoch_c']
    best_test_epoch_c = ckpt['best_test_epoch_c']
    print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, start_epoch-1))   

# set stylegan
gan_path = '/data/stylegan_old/pretrain/stylegan_imagenet.pth'
gan = StyleGANGeneratorModel()
state_dict = torch.load(gan_path)
var_name = 'truncation.truncation'
state_dict[var_name] = gan.state_dict()[var_name]
gan.load_state_dict(state_dict)
gan = gan.synthesis
for p in gan.parameters():
    p.requires_grad_(False)
gan = move_to_device(gan, cfg, device)
model = torch.nn.Sequential(gan, net)
model = model.to(device)

# set dataset, dataloader
dataset = get_dataset(cfg)
transform = get_transform(cfg)

trainset = dataset(root=trainset_cfg.path, train=True, transform=transform.classifier_training)
testset = dataset(root=testset_cfg.path, train=False, transform=transform.classifier_testing)

train_sampler = None
test_sampler = None
if cfg.distributed:
    train_sampler = DistributedSampler(trainset)
    test_sampler = DistributedSampler(testset)

trainloader = DataLoader(trainset,
                         batch_size=trainset_cfg.batch_size,
                         num_workers=4,
                         shuffle=(train_sampler is None),
                         sampler=train_sampler)
testloader = DataLoader(testset,
                        batch_size=testset_cfg.batch_size,
                        num_workers=4,
                        shuffle=False,
                        sampler=test_sampler)

# Normal training
def train(epoch):
    net.train()

    loss_meter_c = AverageMeter()
    acc_meter_c = AverageMeter()
    
    progress_bar = tqdm(trainloader)
    
    for batch_idx, (images, _, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        # FP
        logits_c = net(images)
        
        # Loss, soft_out, preds
        loss_c = criterion(logits_c, labels)
        soft_out_c = torch.softmax(logits_c, dim=1)
        preds_c = soft_out_c.argmax(dim=1)
    
        # BP
        optimizer.zero_grad() # zero the parameter gradients
        loss_c.backward() # backpropagate the loss
        optimizer.step() # update the weights
        
        # Update metrics
        loss_meter_c.update(loss_c.item())
        acc_meter_c.update(((preds_c == labels).float().mean().item() * 100.0))
        
        # Update progress bar
        progress_bar.set_description(f"E: [{epoch}] Train Loss: {loss_meter_c.avg:.4f}, Acc: {acc_meter_c.avg:.3f}") 
    
    return loss_meter_c.avg, acc_meter_c.avg

# Normal testing
def test(epoch):
    
    net.eval()

    acc_meter_c = AverageMeter()
    loss_meter_c = AverageMeter()
    
    progress_bar = tqdm(testloader)
    
    with torch.no_grad():
        for batch_idx, (images, _, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            # FP
            logits_c = net(images)
            
            # Loss, soft_out, preds
            loss_c = criterion(logits_c, labels)
            soft_out_c = torch.softmax(logits_c, dim=1)
            preds_c = soft_out_c.argmax(dim=1)
            
            # Update metrics
            loss_meter_c.update(loss_c.item())
            acc_meter_c.update(((preds_c == labels).float().mean().item() * 100.0))
            
            # Update progress bar
            progress_bar.set_description(f"E: [{epoch}] Test Loss: {loss_meter_c.avg:.4f}, Acc: {acc_meter_c.avg:.3f}")
            
    return loss_meter_c.avg, acc_meter_c.avg

# ----------------------- TESTING WITH ADV, LADV - ROBUSTNESS ATTACKS - ON-OFF-PGD50 -----------------------
# ADV attackers
adv_attack_budgets = [0.02, 0.05] # [0.02, 0.05, 0.1, 0.2, 0.3]

robustness_adv_acc_results = [0 for _ in range(len(adv_attack_budgets))]
robustness_adv_epoch_results = [-1 for _ in range(len(adv_attack_budgets))]

adv_attackers = []
for budget in adv_attack_budgets:
    adv_attackers.append(PGDAttack(predict=net,
                                   eps=budget,
                                   eps_iter=budget/4.0,
                                   nb_iter=50,
                                   clip_min=cfg.image_attack.args.clip_min,
                                   clip_max=cfg.image_attack.args.clip_max))

# LADV attackers
ladv_attack_budgets = [0.02, 0.05] # [0.02, 0.05, 0.1, 0.2, 0.3]

robustness_ladv_acc_results = [0 for _ in range(len(ladv_attack_budgets))]
robustness_ladv_epoch_results = [-1 for _ in range(len(ladv_attack_budgets))]

ladv_attackers = []
for budget in ladv_attack_budgets:
    ladv_attackers.append(PGDAttack(predict=model,
                                    eps=budget,
                                    eps_iter=budget/4.0,
                                    nb_iter=50,
                                    clip_min=None,
                                    clip_max=None))    

def robustness_test_OnOffPGD50(epoch):
    
    dataloader = testloader
    
    progress_bar = tqdm(dataloader)
    net.eval()
    gan.eval()
    
    # Adversarial Image - Image attack
    loss_adv_meters = [AverageMeter() for _ in range(len(adv_attackers))]
    acc_adv_meters = [AverageMeter() for _ in range(len(adv_attackers))]
    
    # Latent Vector based Adversarial Image - Latent attack
    loss_ladv_meters = [AverageMeter() for _ in range(len(ladv_attackers))]
    acc_ladv_meters = [AverageMeter() for _ in range(len(ladv_attackers))]

    for batch_idx, (images, latents, labels) in enumerate(progress_bar):
        images, latents, labels = images.to(device), latents.to(device), labels.to(device)
        
        # Adversarial Image - Image attackers - Robustness Test
        for i, test_attacker in enumerate(adv_attackers):
            with ctx_noparamgrad_and_eval(net):
                images_adv = test_attacker.perturb(images, labels)
                
                # Forward pass
                logits_adv = net(images_adv)
                
                # Calculate loss
                loss_adv = criterion(logits_adv, labels)
                
                # Calculate Predictions
                pred_adv = logits_adv.argmax(dim=1)
                
                # Calculate accuracy
                acc_adv = (pred_adv == labels).float().mean().item() * 100.0

            acc_adv_meters[i].update(acc_adv)
            loss_adv_meters[i].update(loss_adv.item())
        
        # Latent Vector based Adversarial Image - Latent attackers - Robustness Test
        for i, test_latent_attacker in enumerate(ladv_attackers):
            with ctx_noparamgrad_and_eval(model):
                latents_adv = test_latent_attacker.perturb(latents, labels)
                images_ladv = gan(latents_adv).detach()
                
                # Forward pass
                logits_ladv = net(images_ladv)
                
                # Calculate loss
                loss_ladv = criterion(logits_ladv, labels)
                
                # Calculate Predictions
                pred_ladv = logits_ladv.argmax(dim=1)
                
                # Calculate accuracy
                acc_ladv = (pred_ladv == labels).float().mean().item() * 100.0
                
            acc_ladv_meters[i].update(acc_ladv)
            loss_ladv_meters[i].update(loss_ladv.item())    
         
    
    for i, budget in enumerate(adv_attack_budgets):
        wandb.log({f"epoch": epoch, f"test_adv_acc_{budget}": acc_adv_meters[i].avg})
        
        if acc_adv_meters[i].avg > robustness_adv_acc_results[i]:
            robustness_adv_acc_results[i] = acc_adv_meters[i].avg
            robustness_adv_epoch_results[i] = epoch
    
    for i, budget in enumerate(ladv_attack_budgets):
        wandb.log({f"epoch": epoch, f"test_ladv_acc_{budget}": acc_ladv_meters[i].avg})
        
        if acc_ladv_meters[i].avg > robustness_ladv_acc_results[i]:
            robustness_ladv_acc_results[i] = acc_ladv_meters[i].avg
            robustness_ladv_epoch_results[i] = epoch
    
# ----------------------- ROBUSTNESS ATTACKS --------------------------------


for epoch in (range(start_epoch, cfg.num_epochs)):
        
    # Train  
    epoch_train_loss_c, epoch_train_acc_c = train(epoch)
    
    # Test
    epoch_test_loss_c, epoch_test_acc_c = test(epoch)
    
    # Robustness Test
    if cfg.robustness_test:
        robustness_test_OnOffPGD50(epoch)
    
    # lr used
    lr = optimizer.param_groups[0]['lr']
    
    # LR scheduler
    #lr_schedule.step()
    
    # Log it to logging file
    logging.info(f"Epoch: {epoch} | LR: {lr} | Train Loss: {epoch_train_loss_c} | Train Acc: {epoch_train_acc_c} | Test Loss: {epoch_test_loss_c} | Test Acc: {epoch_test_acc_c}")
    
    # Log to wandb
    wandb.log({
        "epoch": epoch,
        "lr": lr,
        "train_loss_c": epoch_train_loss_c,
        "train_acc_c": epoch_train_acc_c,
        "test_loss_c": epoch_test_loss_c,
        "test_acc_c": epoch_test_acc_c
    })
    
    if epoch_train_acc_c > best_train_acc_c:
        best_train_acc_c = epoch_train_acc_c
        best_train_loss_c = epoch_train_loss_c
        best_train_epoch_c = epoch
          
    # Always replace and save the model with best test accuracy
    if epoch_test_acc_c > best_test_acc_c:
        best_test_acc_c = epoch_test_acc_c
        best_test_loss_c = epoch_test_loss_c
        
        # Save the model with {dataset_name}_{}_best_model.pth, where {} is cfg.optimizer.name
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_test_acc_c': best_test_acc_c,
            'best_train_loss_c': best_train_loss_c,
            'best_train_acc_c': best_train_acc_c,
            'best_test_loss_c': best_test_loss_c
        }, os.path.join(output_dir, f'{cfg.dataset.name}_{cfg.optimizer.name}_best_model.pth'))   
    
    # Save the trained model
    checkpoint_path = os.path.join(output_dir, f'{cfg.dataset.name}_classifier_e_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_test_acc_c': best_test_acc_c,
        'best_train_loss_c': best_train_loss_c,
        'best_train_acc_c': best_train_acc_c,
        'best_test_loss_c': best_test_loss_c,
        'best_train_epoch_c': best_train_epoch_c,
        'best_test_epoch_c': best_test_epoch_c
    }, checkpoint_path)
        
# Log best test accuracy and best train loss to wandb        
print(f"Best | Test - Acc: {best_test_acc_c}, Loss: {best_test_loss_c} | Train - Acc: {best_train_acc_c}, Loss: {best_train_loss_c}")

wandb.log({
    "best_test_acc_c": best_test_acc_c,
    "best_test_loss_c": best_test_loss_c,
    "best_train_acc_c": best_train_acc_c,
    "best_train_loss_c": best_train_loss_c,
    "best_train_epoch_c": best_train_epoch_c,
    "best_test_epoch_c": best_test_epoch_c
})

for i, budget in enumerate(adv_attack_budgets):
    wandb.log({
        "adv_budgets": budget,
        "robustness_adv_acc": robustness_adv_acc_results[i],
    })

for i, budget in enumerate(ladv_attack_budgets):
    wandb.log({
        "ladv_budgets": budget,
        "robustness_ladv_acc": robustness_ladv_acc_results[i],
    })
    
wandb.log({
    "adv_budgets": adv_attack_budgets,
    "robustness_acc_results": robustness_adv_acc_results,
    "robustness_epoch_results": robustness_adv_epoch_results,
    "ladv_budgets": ladv_attack_budgets,
    "robustness_ladv_acc_results": robustness_ladv_acc_results,
    "robustness_ladv_epoch_results": robustness_ladv_epoch_results
})

wandb.finish()