import os
import shutil
import logging
from tqdm import tqdm
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
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
from bigmodelvis import Visualization

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        
class BayesWrap(nn.Module):
    def __init__(self, NET):
        super().__init__()

        num_particles = int(cfg.num_particles)
        self.h_kernel = 0
        self.particles = []

        for i in range(num_particles):
            self.particles.append(copy.deepcopy(NET))
            if cfg.add_noise:
                self.particles[i].apply(init_weights)

        for i, particle in enumerate(self.particles):
            self.add_module(str(i), particle)

        # logging.info("num particles: %d" % len(self.particles))
        print(f"num particles: {len(self.particles)}")

    def sample_particle(self):
        return self.particles[np.random.randint(0, len(self.particles))]

    def get_particle(self, index):
        return self.particles[index]

    def get_losses_clean(self, x, labels, criterion, test_mode: bool = False, **kwargs):
        logits, entropies, soft_out, stds, losses = [], [], [], [], []
        return_entropy = "return_entropy" in kwargs and kwargs["return_entropy"]
        if test_mode:
            is_ig = False
        else:
            is_ig = cfg.svgd.is_ig
        x, labels = x.to(device), labels.to(device)
        for particle in self.particles:
            l = particle(x)
            
            loss = criterion(l, labels)
            sft = torch.softmax(l, 1)
            
            logits.append(l)
            losses.append(loss)
            soft_out.append(sft)
            
            if return_entropy:
                l = torch.softmax(l, 1)
                entropies.append((-l * torch.log(l + 1e-8)).sum(1))
            
            if is_ig:
                prob = torch.softmax(l, 1)
                entropies.append((-prob * torch.log(prob + 1e-8)).sum(1))
                
        logits = torch.stack(logits).mean(0)
        ce_loss = torch.stack(losses).mean(0)
        stds = torch.stack(soft_out).std(0)
        soft_out = torch.stack(soft_out).mean(0)
        
        if return_entropy:
            entropies = torch.stack(entropies).mean(0)
            return logits, entropies, soft_out, stds
        
        if is_ig:
            child_entropies = torch.stack(entropies).mean(0)
            parent_prob = torch.softmax(logits, 1)
            parent_entropy = (-parent_prob * torch.log(parent_prob + 1e-8)).sum(1)
            ig_loss = torch.abs(parent_entropy - child_entropies).mean(0)
            
            ig_lambda = cfg.svgd.ig_lambda
            overall_loss = ce_loss + ig_lambda * ig_loss
            wandb.log({"CE Loss": ce_loss, "IG Loss": ig_loss, "Ratio": (ce_loss/ig_loss), "Total Loss": overall_loss})
            return logits, ce_loss, ig_loss, overall_loss, soft_out
            
        return logits, ce_loss, soft_out

    def forward(self, x, **kwargs):
        logits, entropies, soft_out, stds = [], [], [], []
        return_entropy = "return_entropy" in kwargs and kwargs["return_entropy"]
        x = x.to(device)
        for particle in self.particles:
            l = particle(x)
        
            sft = torch.softmax(l, 1)
            logits.append(l)
            soft_out.append(sft)
            
            if return_entropy:
                l = torch.softmax(l, 1)
                entropies.append((-l * torch.log(l + 1e-8)).sum(1))
        
        logits = torch.stack(logits).mean(0)
        stds = torch.stack(soft_out).std(0)
        soft_out = torch.stack(soft_out).mean(0)
        if return_entropy:
            entropies = torch.stack(entropies).mean(0)
            return logits, entropies, soft_out, stds
        return logits

    # New code for SVGD
    def update_grads(self, epoch: int = 0):
        if np.random.rand() < cfg.svgd.p_update:
            return

        all_pgs = self.particles
        no_of_particles = len(all_pgs)
        if self.h_kernel <= 0:
            self.h_kernel = 0.1
        dists = []
        alpha = cfg.svgd.alpha
        new_parameters = [None] * no_of_particles

        for i in range(no_of_particles):
            par1_params = [p for p in all_pgs[i].parameters() if p.grad is not None]
            par1_params_flat = torch.cat([p.data.view(-1) for p in par1_params])
            
            # New Variable to hold parameters for each particle (Whose gradients are recalculated based on other particles' gradients + Repulsion/push)
            new_parameters[i] = {}
            
            # Initialising current particle's parameters with zeros
            for l, p in enumerate(par1_params):
                new_parameters[i][l] = p.grad.data.clone().zero_()
            
            # Compare with respect to all other particles (Including itself)       
            for j in range(no_of_particles):
                par2_params = [p for p in all_pgs[j].parameters() if p.grad is not None]
                par2_params_flat = torch.cat([p.data.view(-1) for p in par2_params])
                
                # check if gradients are None
                if len(par1_params) == 0 or len(par2_params) == 0:
                    continue
                
                # Updating particles
                l2_distance_between_particles = torch.sqrt(((par1_params_flat - par2_params_flat)**2).sum())
                dists.append(l2_distance_between_particles.cpu().item())
                kij = torch.exp(-(l2_distance_between_particles**2)/ self.h_kernel)
                grad_kij = -kij * ((2*l2_distance_between_particles) / self.h_kernel)
                
                # SVGD Update Rule
                driving_force = [(kij/float(no_of_particles))*p2.grad.data for p2 in par2_params]
                repulsive_force = [(grad_kij/float(no_of_particles))*alpha for _ in range(len(par2_params))]
                new_parameters[i] = [new_parameters[i][l] + (driving_force[l] + repulsive_force[l]) for l in range(len(par1_params))]
                
                # Adding Langevin Noise
                if cfg.add_langevin_noise:
                    lr = optimizer.state_dict()['param_groups'][0]['lr']
                    kij_sqrt_part = [torch.sqrt((2*kij.repeat(p.data.nelement()))/(len(all_pgs)*(float(epoch)))).to(device) for p in par1_params]
                    nj = [torch.distributions.Normal(0, 1).sample(kij_sqrt_part[l].shape).to(device) for l in range(len(par1_params))]
                    langevin_noise = [(kij_sqrt_part[l] * nj[l]).view(p.data.shape) for l, p in enumerate(par1_params)]
                    new_parameters[i] = [new_parameters[i][l] + langevin_noise[l] for l in range(len(par1_params))]
                
                    # sqrt of (sum of squares of langevin noise)
                    langevin_noise_flat = torch.cat([langevin_noise[l].view(-1) for l in range(len(par1_params))])
                    l2_langevin_noise = torch.sqrt((langevin_noise_flat**2).sum())
                    wandb.log({
                        "i": i,
                        "j": j,
                        "epoch": epoch,
                        "l2_distance_between_particles": l2_distance_between_particles,
                        "h_kernel": self.h_kernel,
                        "kij": kij,
                        "grad_kij": grad_kij,
                        "langlevin_noise": l2_langevin_noise
                    })
        
        # take power of 2 of median    
        self.h_kernel = ((np.median(dists)**2)/np.log(float(no_of_particles)))
        
        for i in range(len(all_pgs)):
            for l, p in enumerate([p for p in all_pgs[i].parameters() if p.grad is not None]):
                p.grad.data = new_parameters[i][l]
                

# parse command line options
parser = argparse.ArgumentParser(description="On-manifold adv training")
parser.add_argument("--config", default="our_experiments/classifiers/clean/mix10_clean_s3p_s3pig_ens3p.yml")
parser.add_argument("--resume", default="")
args = parser.parse_args()

cfg = load_config(args.config)
trainset_cfg = cfg.dataset.train
testset_cfg = cfg.dataset.test
print(cfg)

# Setup for Wandb
wandb.init(project="CLEAN_CLASSIFIERS_MIX10_R50", config=cfg)
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
device = torch.device(("cuda:"+str(cfg.gpu_id)) if torch.cuda.is_available() else "cpu")
set_device(cfg, device)
set_random_seed(cfg)
cudnn.benchmark = True

# set classifier
net = get_classifier(cfg, cfg.classifier)
net = net.to(device)
# print(f"INITIAL TEST: ", len(list(net.parameters())))

net = BayesWrap(net)
net = net.to(device)
# Visualization(net).structure_graph()
# print(f"INITIAL TEST 2: ", len(list(net.parameters())))

# set optimizers
optimizer = load_optimizer(cfg.optimizer, params=[p for p in net.parameters() if p.requires_grad])

if cfg.scheduler.type == 'cyclic':
    lr_schedule = lambda t: np.interp([t], cfg.scheduler.args.lr_epochs, cfg.scheduler.args.lr_values)[0]
else:
    lr_schedule = None

start_epoch = 0
best_train_ce_loss, best_train_acc, best_test_loss, best_test_acc = 0.0, 0.0, 0.0, 0.0
if args.resume:
    print("=> loading checkpoint '{}'".format(args.resume))
    ckpt = torch.load(args.resume)
    net.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch']+1
    best_train_ce_loss = ckpt['best_train_ce_loss']
    best_train_acc = ckpt['best_train_acc']
    best_test_loss = ckpt['best_test_loss']
    best_test_acc = ckpt['best_test_acc']
    
criterion = torch.nn.CrossEntropyLoss().to(device)

# set dataset, dataloader
dataset = get_dataset(cfg)
transform = get_transform(cfg)
trainset = dataset(root=trainset_cfg.path, train=True)
testset = dataset(root=testset_cfg.path, train=False) 
# Natural image
# testset = test_dataset(root=testset_cfg.path, train=False)

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

def train(epoch):
    
    progress_bar = tqdm(trainloader)

    net.train()

    ce_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    is_ig = cfg.svgd.is_ig
    if is_ig:
        ig_loss_meter = AverageMeter()
        overall_loss_meter = AverageMeter()

    for batch_idx, (images, _, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        lr = cfg.optimizer.args.lr
        if lr_schedule is not None:
            lr = lr_schedule(epoch + (batch_idx + 1) / len(trainloader))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        if is_ig:
            _, ce_loss, ig_loss, overall_loss, soft_out = net.get_losses_clean(images, labels, criterion)
        else:    
            _, overall_loss, soft_out = net.get_losses_clean(images, labels, criterion)
        
        # FP, BP
        optimizer.zero_grad()
        overall_loss.backward()
        if ((cfg.issvgd) and (epoch < cfg.svgd.particle_push_limit_epochs)):
            net.update_grads(epoch)
        optimizer.step()

        preds = soft_out.argmax(dim=1)
        acc_value = (preds == labels).float().mean().item() * 100.0    
        
        if is_ig:
            ce_loss_meter.update(ce_loss.item())
            acc_meter.update(acc_value)
            ig_loss_meter.update(ig_loss.item())
            overall_loss_meter.update(overall_loss.item())
            progress_bar.set_description(f"E: [{epoch}] Train CE Loss: {ce_loss_meter.avg:.4f}, IG Loss: {ig_loss_meter.avg:.4f}, Ov Loss: {overall_loss_meter.avg:.4f}, Acc: {acc_meter.avg:.3f}")
        else:
            ce_loss_meter.update(overall_loss.item())
            acc_meter.update(acc_value)
            progress_bar.set_description(f"E: [{epoch}] Train Loss: {ce_loss_meter.avg:.4f}, Acc: {acc_meter.avg:.3f}")
        
    if is_ig:
        return ce_loss_meter.avg, ig_loss_meter.avg, overall_loss_meter.avg, acc_meter.avg
    else:
        return ce_loss_meter.avg, acc_meter.avg
    

def test(epoch):
    net.eval()

    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    progress_bar = tqdm(testloader)
    
    for batch_idx, (images, _, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        logits, loss, soft_out = net.get_losses_clean(images, labels, criterion, test_mode=True)
        preds = soft_out.argmax(dim=1)
        
        acc_value = (preds == labels).float().mean().item() * 100.0
        loss_meter.update(loss.item())
        acc_meter.update(acc_value)
        
        progress_bar.set_description(f"E: [{epoch}] Test Loss: {loss_meter.avg:.4f}, Acc: {acc_meter.avg:.3f}")
        
    return loss_meter.avg, acc_meter.avg

# ----------------------- TESTING WITH ADV - ROBUSTNESS ATTACKS -----------------------
# ADV attackers
adv_attack_budgets = [0.015, 0.02, 0.035, 0.05, 0.07, 0.1] # TOBEFIXED
robustness_acc_results = [0 for _ in range(len(adv_attack_budgets))]
robustness_epoch_results = [-1 for _ in range(len(adv_attack_budgets))]
adv_attackers = []
for budget in adv_attack_budgets:
    adv_attackers.append(PGDAttack(predict=net,
                                   eps=budget,
                                   eps_iter=budget/4.0,
                                   nb_iter=50,
                                   clip_min=-1.0,
                                   clip_max=1.0)) # Image ranges change for MIX10, CIFAR
    
def robustness_test(epoch):
    
    dataloader = testloader
    
    progress_bar = tqdm(dataloader)
    net.eval()
    
    # Adversarial Image - Image attack
    loss_adv_meters = [AverageMeter() for _ in range(len(adv_attackers))]
    acc_adv_meters = [AverageMeter() for _ in range(len(adv_attackers))]

    for batch_idx, (images, _, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
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
    
    for i, budget in enumerate(adv_attack_budgets):
        wandb.log({f"epoch": epoch, f"test_adv_acc_{budget}": acc_adv_meters[i].avg})
        
        if acc_adv_meters[i].avg > robustness_acc_results[i]:
            robustness_acc_results[i] = acc_adv_meters[i].avg
            robustness_epoch_results[i] = epoch
    
# ----------------------- ROBUSTNESS ATTACKS -----------------------

# ----------------------- L2 Distance Between Particles -----------------------
def calc_L2Dis_between_particles(epoch, log=True):
    ckpt = net.state_dict()
    
    # Assuming ckpt['state_dict'] is your checkpoint's state dictionary
    weights0 = [v for k, v in ckpt.items() if k.startswith('0.')]
    weights1 = [v for k, v in ckpt.items() if k.startswith('1.')]
    weights2 = [v for k, v in ckpt.items() if k.startswith('2.')]

    # Example for calculating L2 distance between weights (Weights - for each pair calculate L2 distance and sum them)
    l2_distance = [0, 0, 0]
    l2_distance[0] = sum((w0 - w1).pow(2).sum() for w0, w1 in zip(weights0, weights1)).sqrt()
    l2_distance[1] = sum((w1 - w2).pow(2).sum() for w1, w2 in zip(weights1, weights2)).sqrt()
    l2_distance[2] = sum((w2 - w0).pow(2).sum() for w2, w0 in zip(weights2, weights0)).sqrt()
    
    if log:
        wandb.log({
            "epoch": epoch,
            "l2_distance_01": l2_distance[0],
            "l2_distance_12": l2_distance[1],
            "l2_distance_20": l2_distance[2]})
    else:
        print(f"Before training started: l2_distance_01: {l2_distance[0]}, l2_distance_12: {l2_distance[1]}, l2_distance_20: {l2_distance[2]}")

# ---------------------------- L2 DISTANCE ----------------------------

for epoch in range(start_epoch, cfg.num_epochs):
    
    # Calculate distance between weights
    if (epoch == 0):
        calc_L2Dis_between_particles(epoch, log=False)
    
    # Train
    if cfg.svgd.is_ig:
        epoch_train_ce_loss, epoch_train_ig_loss, epoch_train_overall_loss, epoch_train_acc = train(epoch)
    else:       
        epoch_train_ce_loss, epoch_train_acc = train(epoch)
    
    # Test
    epoch_test_loss, epoch_test_acc = test(epoch)
    
    # Robustness Test
    robustness_test(epoch)
    
    # lr used
    lr = optimizer.param_groups[0]['lr']
    
    # Calculate distance between particles
    calc_L2Dis_between_particles(epoch)
    
    
    # Log it to logging file
    if cfg.svgd.is_ig:
        logging.info(f"Epoch: {epoch} | LR: {lr} | Train CE Loss: {epoch_train_ce_loss} | IG Loss: {epoch_train_ig_loss} | Overall Loss: {epoch_train_overall_loss} | Acc: {epoch_train_acc} | Test Loss: {epoch_test_loss} | Acc: {epoch_test_acc}")

        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "lr": lr,
            "train_ce_loss": epoch_train_ce_loss,
            "train_ig_loss": epoch_train_ig_loss,
            "train_overall_loss": epoch_train_overall_loss,
            "train_acc": epoch_train_acc,
            "test_loss": epoch_test_loss,
            "test_acc": epoch_test_acc
        })
    else:
        logging.info(f"Epoch: {epoch} | LR: {lr} | Train Loss: {epoch_train_ce_loss} | Train Acc: {epoch_train_acc} | Test Loss: {epoch_test_loss} | Test Acc: {epoch_test_acc}")
            
        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "lr": lr,
            "train_ce_loss": epoch_train_ce_loss,
            "train_acc": epoch_train_acc,
            "test_loss": epoch_test_loss,
            "test_acc": epoch_test_acc
        })
        
    
    if epoch_train_acc > best_train_acc:
        best_train_acc = epoch_train_acc
        best_train_ce_loss = epoch_train_ce_loss
          
    # Always replace and save the model with best test accuracy
    if epoch_test_acc > best_test_acc:
        best_test_acc = epoch_test_acc
        best_test_loss = epoch_test_loss
        
        # Save the model with cifar_{}_best_model.pth, where {} is cfg.optimizer.name
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_test_acc': best_test_acc,
            'best_train_ce_loss': best_train_ce_loss,
            'best_train_acc': best_train_acc,
            'best_test_loss': best_test_loss
        }, os.path.join(output_dir, f'mix10_{cfg.optimizer.name}_best_model.pth'))
        
    # Save the trained model
    checkpoint_path = os.path.join(output_dir, f'classifier_epoch_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_test_acc': best_test_acc,
        'best_train_ce_loss': best_train_ce_loss,
        'best_train_acc': best_train_acc,
        'best_test_loss': best_test_loss
    }, checkpoint_path)
        
# Log best test accuracy and best train loss to wandb        
print(f"Best | Test - Acc: {best_test_acc} | Loss: {best_test_loss} | Train - Acc: {best_train_acc} | Loss: {best_train_ce_loss}")

wandb.log({
    "best_test_acc": best_test_acc,
    "best_test_loss": best_test_loss,
    "best_train_acc": best_train_acc,
    "best_train_ce_loss": best_train_ce_loss
})

wandb.log({
    "budgets": adv_attack_budgets,
    "robustness_acc_results": robustness_acc_results,
    "robustness_epoch_results": robustness_epoch_results
})

wandb.finish()        