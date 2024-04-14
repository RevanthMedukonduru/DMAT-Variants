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

    def get_losses_adv(self, image_c, image_a, labels, criterion, test_mode: bool = False, **kwargs):
        child_clean_logits, child_clean_entropies, child_clean_soft_out, child_clean_losses = [], [], [], []
        child_adv_logits, child_adv_entropies, child_adv_soft_out, child_adv_losses = [], [], [], []
        
        if test_mode:
            is_ig = False
        else:
            is_ig = cfg.svgd.is_ig
        
        image_c, image_a, labels = image_c.to(device), image_a.to(device), labels.to(device)
        for particle in self.particles:
            logits_c = particle(image_c)
            logits_a = particle(image_a)
            
            loss_c = criterion(logits_c, labels)
            loss_a = criterion(logits_a, labels)
            
            sft_c = torch.softmax(logits_c, 1)
            sft_a = torch.softmax(logits_a, 1)
            
            child_clean_logits.append(logits_c)
            child_clean_losses.append(loss_c)
            child_clean_soft_out.append(sft_c)
            
            child_adv_logits.append(logits_a)
            child_adv_losses.append(loss_a)
            child_adv_soft_out.append(sft_a)
                
            if is_ig:
                prob_c = torch.softmax(logits_c, 1)
                prob_a = torch.softmax(logits_a, 1)
                
                child_clean_entropies.append((-prob_c * torch.log(prob_c + 1e-8)).sum(1))
                child_adv_entropies.append((-prob_a * torch.log(prob_a + 1e-8)).sum(1))
            
        child_clean_logits = torch.stack(child_clean_logits).mean(0)
        child_clean_ce_loss = torch.stack(child_clean_losses).mean(0)
        child_clean_soft_out = torch.stack(child_clean_soft_out).mean(0)
        
        child_adv_logits = torch.stack(child_adv_logits).mean(0)
        child_adv_ce_loss = torch.stack(child_adv_losses).mean(0)
        child_adv_soft_out = torch.stack(child_adv_soft_out).mean(0)
        

        if is_ig:
            child_clean_entropies = torch.stack(child_clean_entropies).mean(0)
            child_adv_entropies = torch.stack(child_adv_entropies).mean(0)
            
            parent_prob_clean = torch.softmax(child_clean_logits, 1)
            parent_prob_adv = torch.softmax(child_adv_logits, 1)
            
            parent_entropy_clean = (-parent_prob_clean * torch.log(parent_prob_clean + 1e-8)).sum(1)
            parent_entropy_adv = (-parent_prob_adv * torch.log(parent_prob_adv + 1e-8)).sum(1)
            
            ig_loss_clean = parent_entropy_clean - child_clean_entropies
            ig_loss_adv = parent_entropy_adv - child_adv_entropies
            
            final_ig_loss_adv = torch.abs(ig_loss_clean - ig_loss_adv).mean(0)
            
            ig_lambda = cfg.svgd.ig_lambda
            ce_loss = child_adv_ce_loss  # (0.5 * child_clean_ce_loss + 0.5 * child_adv_ce_loss) - our idea, but DMAT paper used only ADV_CE loss for AT.
            overall_loss = ce_loss + (ig_lambda * final_ig_loss_adv)
            
            wandb.log({"CE Loss": ce_loss, "IG Loss": final_ig_loss_adv, "Ratio": (ce_loss/final_ig_loss_adv), "Total Loss": overall_loss})
            return child_clean_ce_loss, child_adv_ce_loss, final_ig_loss_adv, overall_loss, child_clean_soft_out, child_adv_soft_out
            
        return child_clean_ce_loss, child_adv_ce_loss, child_clean_soft_out, child_adv_soft_out
    

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
parser.add_argument("--config", default="our_experiments/classifiers/adv/mix10_adv_s3p_s3pig_ens3p.yml")
parser.add_argument("--resume", default="")
args = parser.parse_args()

cfg = load_config(args.config)
trainset_cfg = cfg.dataset.train
testset_cfg = cfg.dataset.test
print(cfg)

# Setup for Wandb
wandb.init(project="ADV_CLASSIFIERS_MIX10_R50", config=cfg)
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

net = BayesWrap(net)
net = net.to(device)

# set optimizers
optimizer = load_optimizer(cfg.optimizer, params=[p for p in net.parameters() if p.requires_grad])

if cfg.scheduler.type == 'cyclic':
    lr_schedule = lambda t: np.interp([t], cfg.scheduler.args.lr_epochs, cfg.scheduler.args.lr_values)[0]
else:
    lr_schedule = None

start_epoch = 0
best_train_clean_acc, best_train_clean_loss, best_train_adv_acc, best_train_adv_loss = 0.0, 0.0, 0.0, 0.0
best_test_clean_acc, best_test_clean_loss, best_test_adv_acc, best_test_adv_loss = 0.0, 0.0, 0.0, 0.0
best_train_clean_epoch, best_train_adv_epoch = 0, 0
best_test_clean_epoch, best_test_adv_epoch = 0, 0
if args.resume:
    print("=> loading checkpoint resuming '{}'".format(args.resume))
    ckpt = torch.load(args.resume)
    net.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch']+1
    best_train_clean_acc = ckpt['best_train_clean_acc']
    best_train_clean_loss = ckpt['best_train_clean_loss']
    best_train_adv_acc = ckpt['best_train_adv_acc']
    best_train_adv_loss = ckpt['best_train_adv_loss']
    best_train_clean_epoch = ckpt['best_train_clean_epoch']
    best_train_adv_epoch = ckpt['best_train_adv_epoch']
    best_test_clean_acc = ckpt['best_test_clean_acc']
    best_test_clean_loss = ckpt['best_test_clean_loss']
    best_test_adv_acc = ckpt['best_test_adv_acc']
    best_test_adv_loss = ckpt['best_test_adv_loss']
    best_test_clean_epoch = ckpt['best_test_clean_epoch']
    best_test_adv_epoch = ckpt['best_test_adv_epoch']
    
criterion = torch.nn.CrossEntropyLoss().to(device)

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
gan = move_to_device(gan, cfg)
model = torch.nn.Sequential(gan, net)
model = model.to(device)

# off-manifold attacks/AT
image_attacker = get_attack(cfg.image_attack, net)
test_attacker = PGDAttack(predict=net,
                          eps=cfg.image_attack.args.eps,
                          eps_iter=cfg.image_attack.args.eps_iter,
                          nb_iter=50,
                          clip_min=-1.0,
                          clip_max=1.0)

# set dataset, dataloader
dataset = get_dataset(cfg)
transform = get_transform(cfg)
trainset = dataset(root=trainset_cfg.path, train=True)
testset = dataset(root=testset_cfg.path, train=False) 

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

def train_adv(epoch):
    
    progress_bar = tqdm(trainloader)

    net.train()

    clean_ce_loss_meter = AverageMeter()
    adv_ce_loss_meter = AverageMeter()
    
    clean_acc_meter = AverageMeter()
    adv_acc_meter = AverageMeter()
    
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
        
        with ctx_noparamgrad_and_eval(net):
            images_adv = image_attacker.perturb(images, labels)
    
        if is_ig:
            clean_ce_loss, adv_ce_loss, ig_loss, overall_loss, clean_soft_out, adv_soft_out = net.get_losses_adv(images, images_adv, labels, criterion)
        else:    
            clean_ce_loss, adv_ce_loss, clean_soft_out, adv_soft_out = net.get_losses_adv(images, images_adv, labels, criterion)
            overall_loss = adv_ce_loss
        
        # FP, BP
        optimizer.zero_grad()
        overall_loss.backward()
        if ((cfg.issvgd) and (epoch < cfg.svgd.particle_push_limit_epochs)):
            net.update_grads(epoch)
        optimizer.step()
        
        preds_clean = clean_soft_out.argmax(dim=1)
        acc_clean = (preds_clean == labels).float().mean().item() * 100.0
        preds_adv = adv_soft_out.argmax(dim=1)
        acc_adv = (preds_adv == labels).float().mean().item() * 100.0
        
        clean_ce_loss_meter.update(clean_ce_loss.item())
        adv_ce_loss_meter.update(adv_ce_loss.item())
        clean_acc_meter.update(acc_clean)
        adv_acc_meter.update(acc_adv)
        
        if is_ig:
            ig_loss_meter.update(ig_loss.item())
            overall_loss_meter.update(overall_loss.item())            
            progress_bar.set_description(f"E: [{epoch}] Train CE Loss: {clean_ce_loss_meter.avg:.4f}, Adv CE Loss: {adv_ce_loss_meter.avg:.4f}, IG Loss: {ig_loss_meter.avg:.4f}, Ov Loss: {overall_loss_meter.avg:.4f}, Clean Acc: {clean_acc_meter.avg:.3f}, Adv Acc: {adv_acc_meter.avg:.3f}")
        else:
            progress_bar.set_description(f"E: [{epoch}] Train CE Loss: {clean_ce_loss_meter.avg:.4f}, Adv CE Loss: {adv_ce_loss_meter.avg:.4f}, Clean Acc: {clean_acc_meter.avg:.3f}, Adv Acc: {adv_acc_meter.avg:.3f}")
        
    if is_ig:
        return clean_ce_loss_meter.avg, adv_ce_loss_meter.avg, ig_loss_meter.avg, overall_loss_meter.avg, clean_acc_meter.avg, adv_acc_meter.avg
    else:
        return clean_ce_loss_meter.avg, adv_ce_loss_meter.avg, clean_acc_meter.avg, adv_acc_meter.avg
    

def test_adv(epoch):
    
    progress_bar = tqdm(testloader)
    net.eval()

    # Clean Image
    loss_clean_meter = AverageMeter()
    acc_clean_meter = AverageMeter()
    
    # Adversarial Image - Image attack
    loss_adv_meter = AverageMeter()
    acc_adv_meter = AverageMeter()
    
    
    for batch_idx, (images, _, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        with ctx_noparamgrad_and_eval(net):
            images_adv = test_attacker.perturb(images, labels)
        
        clean_ce_loss, adv_ce_loss, clean_soft_out, adv_soft_out = net.get_losses_adv(images, images_adv, labels, criterion, test_mode=True)
        
        preds_clean = clean_soft_out.argmax(dim=1)
        acc_clean = (preds_clean == labels).float().mean().item() * 100.0
        preds_adv = adv_soft_out.argmax(dim=1)
        acc_adv = (preds_adv == labels).float().mean().item() * 100.0
        
        loss_clean_meter.update(clean_ce_loss.item())
        acc_clean_meter.update(acc_clean)
        loss_adv_meter.update(adv_ce_loss.item())
        acc_adv_meter.update(acc_adv)
        
        progress_bar.set_description(f"E: [{epoch}] Test Clean Loss: {loss_clean_meter.avg:.4f}, C.Acc: {acc_clean_meter.avg:.3f}, Adv Loss: {loss_adv_meter.avg:.4f}, A.Acc: {acc_adv_meter.avg:.3f}")
        
    return loss_clean_meter.avg, acc_clean_meter.avg, loss_adv_meter.avg, acc_adv_meter.avg


# ----------------------- TESTING WITH ADV - ROBUSTNESS ATTACKS -----------------------
# ADV attackers
adv_attack_budgets = [0.035, 0.055, 0.07]
ladv_attack_budgets = [0.01, 0.02, 0.03]

robustness_adv_acc_results = [0 for _ in range(len(adv_attack_budgets))]
robustness_adv_epoch_results = [-1 for _ in range(len(adv_attack_budgets))]
robustness_ladv_acc_results = [0 for _ in range(len(ladv_attack_budgets))]
robustness_ladv_epoch_results = [-1 for _ in range(len(ladv_attack_budgets))]

adv_attackers = []
for budget in adv_attack_budgets:
    adv_attackers.append(PGDAttack(predict=net,
                                   eps=budget,
                                   eps_iter=budget/4.0,
                                   nb_iter=50,
                                   clip_min=-1.0,
                                   clip_max=1.0)) # Image ranges change for MIX10, CIFAR

# LADV attackers
ladv_attackers = []
for budget in ladv_attack_budgets:
    ladv_attackers.append(PGDAttack(predict=model,
                                    eps=budget,
                                    eps_iter=budget/4.0,
                                    nb_iter=50,
                                    clip_min=None,
                                    clip_max=None))    

def robustness_test(epoch):
    
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
    if (epoch == 0 and cfg.num_particles>1):
        calc_L2Dis_between_particles(epoch, log=False)
    
    # Train
    if cfg.svgd.is_ig:
        epoch_train_clean_ce_loss, epoch_train_adv_ce_loss, epoch_train_ig_loss, epoch_train_overall_loss, epoch_train_clean_acc, epoch_train_adv_acc = train_adv(epoch)
    else:       
        epoch_train_clean_ce_loss, epoch_train_adv_ce_loss, epoch_train_clean_acc, epoch_train_adv_acc = train_adv(epoch)
    
    # Test
    epoch_test_clean_loss, epoch_test_clean_acc, epoch_test_adv_loss, epoch_test_adv_acc = test_adv(epoch)
    
    # Robustness Test
    if cfg.robustness_test:
        robustness_test(epoch)
    
    # lr used
    lr = optimizer.param_groups[0]['lr']
    
    # Calculate distance between particles
    if (cfg.num_particles>1):
        calc_L2Dis_between_particles(epoch)
    
    
    # Log it to logging file
    if cfg.svgd.is_ig:
        logging.info(f"Epoch: {epoch} | LR: {lr} | Train CE Loss: {epoch_train_clean_ce_loss} | Adv Loss: {epoch_train_adv_ce_loss} | IG Loss: {epoch_train_ig_loss} | Ov Loss: {epoch_train_overall_loss} | Clean Acc: {epoch_train_clean_acc} | Adv Acc: {epoch_train_adv_acc} | Test Clean Loss: {epoch_test_clean_loss} | Adv Loss: {epoch_test_adv_loss} | Clean Acc: {epoch_test_clean_acc} | Adv Acc: {epoch_test_adv_acc}")

        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "lr": lr,
            "train_clean_ce_loss": epoch_train_clean_ce_loss,
            "train_adv_ce_loss": epoch_train_adv_ce_loss,
            "train_ig_loss": epoch_train_ig_loss,
            "train_overall_loss": epoch_train_overall_loss,
            "train_clean_acc": epoch_train_clean_acc,
            "train_adv_acc": epoch_train_adv_acc,
            "test_clean_loss": epoch_test_clean_loss,
            "test_adv_loss": epoch_test_adv_loss,
            "test_clean_acc": epoch_test_clean_acc,
            "test_adv_acc": epoch_test_adv_acc
        })
    else:
        logging.info(f"Epoch: {epoch} | LR: {lr} | Train CE Loss: {epoch_train_clean_ce_loss} | Adv Loss: {epoch_train_adv_ce_loss} | Clean Acc: {epoch_train_clean_acc} | Adv Acc: {epoch_train_adv_acc} | Test Clean Loss: {epoch_test_clean_loss} | Adv Loss: {epoch_test_adv_loss} | Clean Acc: {epoch_test_clean_acc} | Adv Acc: {epoch_test_adv_acc}")
                    
        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "lr": lr,
            "train_clean_ce_loss": epoch_train_clean_ce_loss,
            "train_adv_ce_loss": epoch_train_adv_ce_loss,
            "train_clean_acc": epoch_train_clean_acc,
            "train_adv_acc": epoch_train_adv_acc,
            "test_clean_loss": epoch_test_clean_loss,
            "test_adv_loss": epoch_test_adv_loss,
            "test_clean_acc": epoch_test_clean_acc,
            "test_adv_acc": epoch_test_adv_acc
        })
    
    if epoch_train_clean_acc > best_train_clean_acc:
        best_train_clean_acc = epoch_train_clean_acc
        best_train_clean_loss = epoch_train_clean_ce_loss
        best_train_clean_epoch = epoch
    
    if epoch_train_adv_acc > best_train_adv_acc:
        best_train_adv_acc = epoch_train_adv_acc
        best_train_adv_loss = epoch_train_adv_ce_loss
        best_train_adv_epoch = epoch
        
    if epoch_test_clean_acc > best_test_clean_acc:
        best_test_clean_acc = epoch_test_clean_acc
        best_test_clean_loss = epoch_test_clean_loss
        best_test_clean_epoch = epoch
        
        checkpoint_path = os.path.join(output_dir, f'best_clean_classifier.pt')
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_test_clean_acc': best_test_clean_acc,
            'best_test_clean_loss': best_test_clean_loss,
            'best_test_clean_epoch': best_test_clean_epoch
        }, checkpoint_path)
        
    if epoch_test_adv_acc > best_test_adv_acc:
        best_test_adv_acc = epoch_test_adv_acc
        best_test_adv_loss = epoch_test_adv_loss
        best_test_adv_epoch = epoch
        
        checkpoint_path = os.path.join(output_dir, f'best_adv_classifier.pt')
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_test_adv_acc': best_test_adv_acc,
            'best_test_adv_loss': best_test_adv_loss,
            'best_test_adv_epoch': best_test_adv_epoch
        }, checkpoint_path)
        
    # Save the model with cifar_{}_best_model.pth, where {} is cfg.optimizer.name
    checkpoint_path = os.path.join(output_dir, f'classifier_epoch_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_train_clean_acc': best_train_clean_acc,
        'best_train_clean_loss': best_train_clean_loss,
        'best_train_adv_acc': best_train_adv_acc,
        'best_train_adv_loss': best_train_adv_loss,
        'best_train_clean_epoch': best_train_clean_epoch,
        'best_train_adv_epoch': best_train_adv_epoch,
        'best_test_clean_acc': best_test_clean_acc,
        'best_test_clean_loss': best_test_clean_loss,
        'best_test_adv_acc': best_test_adv_acc,
        'best_test_adv_loss': best_test_adv_loss,
        'best_test_clean_epoch': best_test_clean_epoch,
        'best_test_adv_epoch': best_test_adv_epoch
    }, checkpoint_path)

        
# Log best test accuracy and best train loss to wandb        
print(f"Best | Train: Clean - Acc: {best_train_clean_acc}, C.Loss: {best_train_clean_loss} | Adv - Acc: {best_train_adv_acc}, A.Loss: {best_train_adv_loss}")
print(f"Best | Test: Clean - Acc: {best_test_clean_acc}, C.Loss: {best_test_clean_loss} | Adv - Acc: {best_test_adv_acc}, A.Loss: {best_test_adv_loss}")

wandb.log({
    "best_train_clean_acc": best_train_clean_acc,
    "best_train_clean_loss": best_train_clean_loss,
    "best_train_adv_acc": best_train_adv_acc,
    "best_train_adv_loss": best_train_adv_loss,
    "best_train_clean_epoch": best_train_clean_epoch,
    "best_train_adv_epoch": best_train_adv_epoch,
    "best_test_clean_acc": best_test_clean_acc,
    "best_test_clean_loss": best_test_clean_loss,
    "best_test_adv_acc": best_test_adv_acc,
    "best_test_adv_loss": best_test_adv_loss,
    "best_test_clean_epoch": best_test_clean_epoch,
    "best_test_adv_epoch": best_test_adv_epoch
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