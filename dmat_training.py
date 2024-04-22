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
parser = argparse.ArgumentParser(description="Training using ImageNet-10 - SINGLE NET - DMAT")
parser.add_argument("--config", default="our_experiments/classifiers/single_net/dmat/dmat_config.yml")
parser.add_argument("--resume", default=None)
args = parser.parse_args()

cfg = load_config(args.config)
print(cfg)

trainset_cfg = cfg.dataset.train
testset_cfg = cfg.dataset.test

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

# check if IG objective is used
is_ig = cfg.is_ig

# LR scheduler
# lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)
 
start_epoch = 0
best_train_acc_c, best_train_loss_c, best_train_acc_a, best_train_loss_a, best_train_acc_la, best_train_loss_la = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
best_test_acc_c, best_test_loss_c, best_test_acc_a, best_test_loss_a, best_test_acc_la, best_test_loss_la = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
best_train_epoch_c, best_train_epoch_a, best_train_epoch_la = 0, 0, 0
best_test_epoch_c, best_test_epoch_a, best_test_epoch_la = 0, 0, 0
if args.resume:
    print("=> loading checkpoint resuming '{}'".format(args.resume))
    ckpt = torch.load(args.resume)
    net.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch']+1
    
    best_train_acc_c = ckpt['best_train_acc_c']
    best_train_loss_c = ckpt['best_train_loss_c']
    best_train_epoch_c = ckpt['best_train_epoch_c']
    
    best_test_acc_c = ckpt['best_test_acc_c']
    best_test_loss_c = ckpt['best_test_loss_c']
    best_test_epoch_c = ckpt['best_test_epoch_c']
    
    best_train_acc_a = ckpt['best_train_acc_a']
    best_train_loss_a = ckpt['best_train_loss_a']
    best_train_epoch_a = ckpt['best_train_epoch_a']
    
    best_test_acc_a = ckpt['best_test_acc_a']
    best_test_loss_a = ckpt['best_test_loss_a']
    best_test_epoch_a = ckpt['best_test_epoch_a']
    
    best_train_acc_la = ckpt['best_train_acc_la'] 
    best_train_loss_la = ckpt['best_train_loss_la']
    best_train_epoch_la = ckpt['best_train_epoch_la']
    
    best_test_acc_la = ckpt['best_test_acc_la']
    best_test_loss_la = ckpt['best_test_loss_la']
    best_test_epoch_la = ckpt['best_test_epoch_la']
    
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

# DMAT attacks (On, Off - Manifold)
adv_attacker = get_attack(cfg.image_attack, net)
test_adv_attacker = PGDAttack(predict=net,
                          eps=cfg.image_attack.args.eps,
                          eps_iter=cfg.image_attack.args.eps_iter,
                          nb_iter=50,
                          clip_min=cfg.image_attack.args.clip_min,
                          clip_max=cfg.image_attack.args.clip_max)

latent_attacker = get_attack(cfg.latent_attack, model)
test_latent_attacker = PGDAttack(predict=model,
                                 eps=cfg.latent_attack.args.eps,
                                 eps_iter=cfg.latent_attack.args.eps_iter,
                                 nb_iter=50, 
                                 clip_max=None, clip_min=None)


# Adv training
def train_dmat(epoch):
    net.train()

    loss_meter_c = AverageMeter()
    acc_meter_c = AverageMeter()
    
    loss_meter_a = AverageMeter()
    acc_meter_a = AverageMeter()
    
    loss_meter_la = AverageMeter()
    acc_meter_la = AverageMeter()
    
    if is_ig:
        ig_loss_meter_a = AverageMeter()
        ig_loss_meter_la = AverageMeter()
        
        ig_loss_meter = AverageMeter()
        overall_loss_meter = AverageMeter()
        
    progress_bar = tqdm(trainloader)
    
    for batch_idx, (images, latents, labels) in enumerate(progress_bar):
        images, latents, labels = images.to(device), latents.to(device), labels.to(device)
        
        # FP (Since clean loss is not included in CE Loss of Adv train - we dont need computational graph)
        with torch.no_grad():
            logits_c = net(images)
            
        with ctx_noparamgrad_and_eval(model):
            images_adv = adv_attacker.perturb(images, labels)
            latents_adv = latent_attacker.perturb(latents, labels)
            images_ladv = gan(latents_adv).detach() 
        
        logits_a = net(images_adv)
        logits_la = net(images_ladv)
        
        # Loss, soft_out, preds - Clean, Adv, Latent Adv
        loss_c = criterion(logits_c, labels)
        soft_out_c = torch.softmax(logits_c, dim=1)
        preds_c = soft_out_c.argmax(dim=1)
        
        loss_a = criterion(logits_a, labels)
        soft_out_a = torch.softmax(logits_a, dim=1)
        preds_a = soft_out_a.argmax(dim=1)

        loss_la = criterion(logits_la, labels)
        soft_out_la = torch.softmax(logits_la, dim=1)
        preds_la = soft_out_la.argmax(dim=1)
        
        overall_loss = 0.5*loss_a + 0.5*loss_la # Based on DMAT paper (50% of CE loss of Adv and 50% of CE Loss of Latent Adv)
        
        # IG - Calculations
        if is_ig:
            # Entropy calculation for clean and adversarial logits
            entropy_c = (-soft_out_c * torch.log(soft_out_c + 1e-8)).sum(dim=1)
            entropy_a = (-soft_out_a * torch.log(soft_out_a + 1e-8)).sum(dim=1)
            entropy_la = (-soft_out_la * torch.log(soft_out_la + 1e-8)).sum(dim=1)
            
            # IG Loss for (clean vs adv), (clean vs latent adv)
            ig_loss_a = torch.abs(entropy_c - entropy_a).mean(0)
            ig_loss_la = torch.abs(entropy_c - entropy_la).mean(0)
            
            lambda_a = cfg.ig.lambda_a
            lambda_la = cfg.ig.lambda_la
            ig_loss = lambda_a * ig_loss_a + lambda_la * ig_loss_la
            
            # Overall Loss: CE_loss_adv + Lambda * IG_loss
            lambda_ig = cfg.ig.lambda_ig
            overall_loss = overall_loss + (lambda_ig * ig_loss)

            # update_metrics
            ig_loss_meter_a.update(ig_loss_a.item())
            ig_loss_meter_la.update(ig_loss_la.item())
            ig_loss_meter.update(ig_loss.item())
            overall_loss_meter.update(overall_loss.item())
            
        # BP
        optimizer.zero_grad() # zero the parameter gradients
        overall_loss.backward() # backpropagate the loss
        optimizer.step() # update the weights
        
        # Update metrics
        loss_meter_c.update(loss_c.item())
        acc_meter_c.update(((preds_c == labels).float().mean().item() * 100.0))
        
        loss_meter_a.update(loss_a.item())
        acc_meter_a.update(((preds_a == labels).float().mean().item() * 100.0))
        
        loss_meter_la.update(loss_la.item())
        acc_meter_la.update(((preds_la == labels).float().mean().item() * 100.0))
        
        # Update progress bar
        if is_ig:
            progress_bar.set_description(f"E: [{epoch}] Train C.Loss: {loss_meter_c.avg:.4f}, C.Acc: {acc_meter_c.avg:.3f}, A.Loss: {loss_meter_a.avg:.4f}, A.Acc: {acc_meter_a.avg:.3f}, LA.Loss: {loss_meter_la.avg:.4f}, LA.Acc: {acc_meter_la.avg:.3f}, A.IG.Loss: {ig_loss_meter_a.avg:.4f}, LA.IG.Loss: {ig_loss_meter_la.avg:.4f}, IG.Loss: {ig_loss_meter.avg:.4f}, Ov.Loss: {overall_loss_meter.avg:.4f}")
        else:
            progress_bar.set_description(f"E: [{epoch}] Train C.Loss: {loss_meter_c.avg:.4f}, C.Acc: {acc_meter_c.avg:.3f}, A.Loss: {loss_meter_a.avg:.4f}, A.Acc: {acc_meter_a.avg:.3f}, LA.Loss: {loss_meter_la.avg:.4f}, LA.Acc: {acc_meter_la.avg:.3f}")
    
    if is_ig:
        wandb.log({
            "ig_loss_a": ig_loss_meter_a.avg,
            "ig_loss_la": ig_loss_meter_la.avg
        })
            
    if is_ig:
        return loss_meter_c.avg, acc_meter_c.avg, loss_meter_a.avg, acc_meter_a.avg, loss_meter_la.avg, acc_meter_la.avg, ig_loss_meter.avg, overall_loss_meter.avg
    else:
        return loss_meter_c.avg, acc_meter_c.avg, loss_meter_a.avg, acc_meter_a.avg, loss_meter_la.avg, acc_meter_la.avg

    
# Clean, Adv testing
def test_dmat(epoch):
    
    net.eval()
    gan.eval()

    acc_meter_c = AverageMeter()
    loss_meter_c = AverageMeter()
    
    acc_meter_a = AverageMeter()
    loss_meter_a = AverageMeter()
    
    acc_meter_la = AverageMeter()
    loss_meter_la = AverageMeter()
    
    progress_bar = tqdm(testloader)
    
    for batch_idx, (images, latents, labels) in enumerate(progress_bar):
        images, latents, labels = images.to(device), latents.to(device), labels.to(device)
        
        with ctx_noparamgrad_and_eval(model):
            images_adv = test_adv_attacker.perturb(images, labels)
            latents_ladv = test_latent_attacker.perturb(latents, labels)
            images_ladv = gan(latents_ladv).detach()
        
        # FP
        with torch.no_grad():   
            logits_c = net(images) 
            logits_a = net(images_adv)
            logits_la = net(images_ladv)
        
        # Loss, soft_out, preds
        loss_c = criterion(logits_c, labels)
        soft_out_c = torch.softmax(logits_c, dim=1)
        preds_c = soft_out_c.argmax(dim=1)
        
        loss_a = criterion(logits_a, labels)
        soft_out_a = torch.softmax(logits_a, dim=1)
        preds_a = soft_out_a.argmax(dim=1)
        
        loss_la = criterion(logits_la, labels)
        soft_out_la = torch.softmax(logits_la, dim=1)
        preds_la = soft_out_la.argmax(dim=1)
        
        # Update metrics
        loss_meter_c.update(loss_c.item())
        acc_meter_c.update(((preds_c == labels).float().mean().item() * 100.0))
        
        loss_meter_a.update(loss_a.item())
        acc_meter_a.update(((preds_a == labels).float().mean().item() * 100.0))
        
        loss_meter_la.update(loss_la.item())
        acc_meter_la.update(((preds_la == labels).float().mean().item() * 100.0))
        
        # Update progress bar
        progress_bar.set_description(f"E: [{epoch}] Test C.Loss: {loss_meter_c.avg:.4f}, C.Acc: {acc_meter_c.avg:.3f}, A.Loss: {loss_meter_a.avg:.4f}, A.Acc: {acc_meter_a.avg:.3f}, LA.Loss: {loss_meter_la.avg:.4f}, LA.Acc: {acc_meter_la.avg:.3f}")
        
    return loss_meter_c.avg, acc_meter_c.avg, loss_meter_a.avg, acc_meter_a.avg, loss_meter_la.avg, acc_meter_la.avg

# ----------------------- TESTING WITH ADV, LADV - ROBUSTNESS ATTACKS - ON-OFF-PGD50 -----------------------
# ADV attackers
adv_attack_budgets = [0.02, 0.05] #[0.02, 0.05, 0.1, 0.2, 0.3]

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
ladv_attack_budgets = [0.02, 0.05] #[0.02, 0.05, 0.1, 0.2, 0.3]

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
    if is_ig:
        epoch_train_loss_c, epoch_train_acc_c, epoch_train_loss_a, epoch_train_acc_a, epoch_train_loss_la, epoch_train_acc_la, epoch_train_ig_loss, epoch_train_overall_loss = train_dmat(epoch)
    else:
        epoch_train_loss_c, epoch_train_acc_c, epoch_train_loss_a, epoch_train_acc_a, epoch_train_loss_la, epoch_train_acc_la = train_dmat(epoch)
    
    # Test
    epoch_test_loss_c, epoch_test_acc_c, epoch_test_loss_a, epoch_test_acc_a, epoch_test_loss_la, epoch_test_acc_la = test_dmat(epoch)
    
    # Robustness Test
    if cfg.robustness_test:
        robustness_test_OnOffPGD50(epoch)
    
    # lr used
    lr = optimizer.param_groups[0]['lr']
    
    # LR scheduler
    #lr_schedule.step()
    
    # Log it to logging file
    logging.info(f"Epoch: {epoch} | LR: {lr} | Train C.Loss: {epoch_train_loss_c}, C.Acc: {epoch_train_acc_c}, A.Loss: {epoch_train_loss_a}, A.Acc: {epoch_train_acc_a}, LA.Loss: {epoch_train_loss_la}, LA.Acc: {epoch_train_acc_la} | Test C.Loss: {epoch_test_loss_c}, C.Acc: {epoch_test_acc_c}, A.Loss: {epoch_test_loss_a}, A.Acc: {epoch_test_acc_a}, LA.Loss: {epoch_test_loss_la}, LA.Acc: {epoch_test_acc_la}")
    
    # Log to wandb
    wandb.log({
        "epoch": epoch,
        "lr": lr,
        "train_loss_c": epoch_train_loss_c,
        "train_acc_c": epoch_train_acc_c,
        "train_loss_a": epoch_train_loss_a,
        "train_acc_a": epoch_train_acc_a,
        "train_loss_la": epoch_train_loss_la,
        "train_acc_la": epoch_train_acc_la,
        "test_loss_c": epoch_test_loss_c,
        "test_acc_c": epoch_test_acc_c,
        "test_loss_a": epoch_test_loss_a,
        "test_acc_a": epoch_test_acc_a,
        "test_loss_la": epoch_test_loss_la,
        "test_acc_la": epoch_test_acc_la
    })
    
    if epoch_train_acc_c > best_train_acc_c:
        best_train_acc_c = epoch_train_acc_c
        best_train_loss_c = epoch_train_loss_c
        best_train_epoch_c = epoch
    
    if epoch_train_acc_a > best_train_acc_a:
        best_train_acc_a = epoch_train_acc_a
        best_train_loss_a = epoch_train_loss_a
        best_train_epoch_a = epoch
    
    if epoch_train_acc_la > best_train_acc_la:
        best_train_acc_la = epoch_train_acc_la
        best_train_loss_la = epoch_train_loss_la
        best_train_epoch_la = epoch
    
    # Replace and save the model with best test adv accuracy
    if epoch_test_acc_a > best_test_acc_a:
        best_test_acc_a = epoch_test_acc_a
        best_test_loss_a = epoch_test_loss_a
        best_test_epoch_a = epoch
        
        # Save the model with {dataset_name}_{}_best_model.pth, where {} is cfg.optimizer.name
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_train_loss_c': best_train_loss_c,
            'best_train_acc_c': best_train_acc_c,
            'best_train_epoch_c': best_train_epoch_c,
            'best_train_loss_a': best_train_loss_a,
            'best_train_acc_a': best_train_acc_a,
            'best_train_epoch_a': best_train_epoch_a,
            'best_test_loss_a': best_test_loss_a,
            'best_test_acc_a': best_test_acc_a,
            'best_test_epoch_a': best_test_epoch_a
        }, os.path.join(output_dir, f'{cfg.dataset.name}_{cfg.optimizer.name}_adv_best_model.pth'))
    
    # Replace and save the model with best test latent adv accuracy
    if epoch_test_acc_la > best_test_acc_la:
        best_test_acc_la = epoch_test_acc_la
        best_test_loss_la = epoch_test_loss_la
        best_test_epoch_la = epoch
        
        # Save the model with {dataset_name}_{}_best_model.pth, where {} is cfg.optimizer.name
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_train_loss_a': best_train_loss_a,
            'best_train_acc_a': best_train_acc_a,
            'best_train_epoch_a': best_train_epoch_a,
            'best_train_acc_la': best_train_acc_la,
            'best_train_loss_la': best_train_loss_la,
            'best_train_epoch_la': best_train_epoch_la,
            'best_test_loss_a': best_test_loss_a,
            'best_test_acc_a': best_test_acc_a,
            'best_test_epoch_a': best_test_epoch_a,
            'best_test_loss_la': best_test_loss_la,
            'best_test_acc_la': best_test_acc_la,
            'best_test_epoch_la': best_test_epoch_la
        }, os.path.join(output_dir, f'{cfg.dataset.name}_{cfg.optimizer.name}_latent_adv_best_model.pth'))
          
    # Replace and save the model with best test clean accuracy
    if epoch_test_acc_c > best_test_acc_c:
        best_test_acc_c = epoch_test_acc_c
        best_test_loss_c = epoch_test_loss_c
        best_test_epoch_c = epoch
        
        # Save the model with {dataset_name}_{}_best_model.pth, where {} is cfg.optimizer.name
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_train_loss_c': best_train_loss_c,
            'best_train_acc_c': best_train_acc_c,
            'best_train_epoch_c': best_train_epoch_c,
            'best_train_loss_a': best_train_loss_a,
            'best_train_acc_a': best_train_acc_a,
            'best_train_epoch_a': best_train_epoch_a,
            'best_train_acc_la': best_train_acc_la,
            'best_train_loss_la': best_train_loss_la,
            'best_train_epoch_la': best_train_epoch_la,
            'best_test_loss_c': best_test_loss_c,
            'best_test_acc_c': best_test_acc_c,
            'best_test_epoch_c': best_test_epoch_c,
            'best_test_loss_a': best_test_loss_a,
            'best_test_acc_a': best_test_acc_a,
            'best_test_epoch_a': best_test_epoch_a,
            'best_test_loss_la': best_test_loss_la,
            'best_test_acc_la': best_test_acc_la,
            'best_test_epoch_la': best_test_epoch_la
        }, os.path.join(output_dir, f'{cfg.dataset.name}_{cfg.optimizer.name}_clean_best_model.pth'))   
    
    # Save the trained model
    checkpoint_path = os.path.join(output_dir, f'{cfg.dataset.name}_classifier_e_{epoch}.pt')
    torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_train_loss_c': best_train_loss_c,
            'best_train_acc_c': best_train_acc_c,
            'best_train_epoch_c': best_train_epoch_c,
            'best_train_loss_a': best_train_loss_a,
            'best_train_acc_a': best_train_acc_a,
            'best_train_epoch_a': best_train_epoch_a,
            'best_train_acc_la': best_train_acc_la,
            'best_train_loss_la': best_train_loss_la,
            'best_train_epoch_la': best_train_epoch_la,
            'best_test_loss_c': best_test_loss_c,
            'best_test_acc_c': best_test_acc_c,
            'best_test_epoch_c': best_test_epoch_c,
            'best_test_loss_a': best_test_loss_a,
            'best_test_acc_a': best_test_acc_a,
            'best_test_epoch_a': best_test_epoch_a,
            'best_test_loss_la': best_test_loss_la,
            'best_test_acc_la': best_test_acc_la,
            'best_test_epoch_la': best_test_epoch_la
    }, checkpoint_path)
        
# Log best test accuracy and best train loss to wandb        
print(f"Best | Train - C.Acc: {best_train_acc_c}, C.Loss: {best_train_loss_c}, A.Acc: {best_train_acc_a}, A.Loss: {best_train_loss_a}, LA.Acc: {best_train_acc_la}, LA.Loss: {best_train_loss_la}")
print(f"Best | Test - C.Acc: {best_test_acc_c}, C.Loss: {best_test_loss_c}, A.Acc: {best_test_acc_a}, A.Loss: {best_test_loss_a}, LA.Acc: {best_test_acc_la}, LA.Loss: {best_test_loss_la}") 

wandb.log({
    "best_test_acc_c": best_test_acc_c,
    "best_test_loss_c": best_test_loss_c,
    "best_test_epoch_c": best_test_epoch_c,
    "best_test_acc_a": best_test_acc_a,
    "best_test_loss_a": best_test_loss_a,
    "best_test_epoch_a": best_test_epoch_a,
    "best_test_acc_la": best_test_acc_la,
    "best_test_loss_la": best_test_loss_la,
    "best_test_epoch_la": best_test_epoch_la,
    "best_train_acc_c": best_train_acc_c,
    "best_train_loss_c": best_train_loss_c,
    "best_train_epoch_c": best_train_epoch_c,
    "best_train_acc_a": best_train_acc_a,
    "best_train_loss_a": best_train_loss_a,
    "best_train_epoch_a": best_train_epoch_a,
    "best_train_acc_la": best_train_acc_la,
    "best_train_loss_la": best_train_loss_la,
    "best_train_epoch_la": best_train_epoch_la
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