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

    # Use get_losses_adv for Adv training (Adv attacks/AT only) - Change in train/test functions accordingly
    def get_losses_adv(self, image_c, image_a, labels, criterion, **kwargs):
        img_losses_c, img_losses_a, img_c_soft_out, img_adv_soft_out = [], [], [], []
        image_c, image_a, labels = image_c.to(device), image_a.to(device), labels.to(device)
        for particle in self.particles:
            logits_c = particle(image_c)
            logits_a = particle(image_a)
            img_loss_c = criterion(logits_c, labels)
            img_loss_a = criterion(logits_a, labels)
            img_losses_c.append(img_loss_c)
            img_losses_a.append(img_loss_a)
            img_c_soft_out.append(torch.softmax(logits_c, 1))
            img_adv_soft_out.append(torch.softmax(logits_a, 1))
        final_img_losses_c = torch.stack(img_losses_c).mean(0)
        final_img_losses_a = torch.stack(img_losses_a).mean(0)
        final_img_soft_out_c = torch.stack(img_c_soft_out).mean(0)
        final_img_soft_out_a = torch.stack(img_adv_soft_out).mean(0)
        return final_img_losses_c, final_img_losses_a, final_img_soft_out_c, final_img_soft_out_a

    # Use get_losses_clean for normal training (No attacks) - Change in train/test functions accordingly
    def get_losses_clean(self, image_c, labels, criterion, **kwargs):
        logits, entropies, soft_out, stds, losses = [], [], [], [], []
        return_entropy = "return_entropy" in kwargs and kwargs["return_entropy"]
        image_c, labels = image_c.to(device), labels.to(device)
        for particle in self.particles:
            l = particle(image_c)
            
            loss = criterion(l, labels)
            sft = torch.softmax(l, 1)
            
            logits.append(l)
            losses.append(loss)
            soft_out.append(sft)
            
            if return_entropy:
                l = torch.softmax(l, 1)
                entropies.append((-l * torch.log(l + 1e-8)).sum(1))
        
        logits = torch.stack(logits).mean(0)
        overall_loss = torch.stack(losses).mean(0)
        stds = torch.stack(soft_out).std(0)
        soft_out = torch.stack(soft_out).mean(0)
        if return_entropy:
            entropies = torch.stack(entropies).mean(0)
            return logits, entropies, soft_out, stds
        return logits, overall_loss, soft_out

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
    
    # New Code
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
                if cfg.add_langevin_noise and (i!=j) and epoch>=1:
                    lr = optimizer.state_dict()['param_groups'][0]['lr']
                    kij_sqrt_part = [torch.sqrt((2*kij.repeat(p.data.nelement()))/(len(all_pgs)*(float(lr)))).to(device) for p in par1_params]
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
                
    # Old Code           
    # def update_grads(self):
    #     if np.random.rand() < cfg.svgd.p_update:
    #         return

    #     all_pgs = self.particles
    #     if self.h_kernel <= 0:
    #         self.h_kernel = 0.1
    #     dists = []
    #     alpha = cfg.svgd.alpha
    #     new_parameters = [None] * len(all_pgs)

    #     for i in range(len(all_pgs)):
    #         # New Variable to hold parameters for each particle (Whose gradients are recalculated based on other particles' gradients + Repulsion/push)
    #         new_parameters[i] = {}
            
    #         # Initialising current particle's parameters with zeros
    #         for l, p in enumerate(all_pgs[i].parameters()):
    #             if p.grad is None:
    #                 new_parameters[i][l] = None
    #             else:
    #                 new_parameters[i][l] = p.grad.data.new(p.grad.data.size()).zero_()
            
    #         # Compare with respect to all other particles (Including itself)       
    #         for j in range(len(all_pgs)):
    #             # Updating each param/weight of the current particle
    #             for l, params in enumerate(zip(all_pgs[i].parameters(), all_pgs[j].parameters())):
    #                 p, p2 = params
    #                 if p.grad is None or p2.grad is None:
    #                     continue
                    
    #                 # ----------  RBF Kernel, SVGD calculations ----------
    #                 # Distance between the parameters of the current particle and the other particle (L2)
    #                 d = (p.data - p2.data).norm(2)
    #                 dists.append(d.cpu().item())
    #                 kij = torch.exp(-(d**2) / self.h_kernel**2 / 2)
    #                 grad_kij = -kij * (d / self.h_kernel**2)
                    
    #                 # SVGD Update Rule
    #                 driving_force = (kij*p2.grad.data)
    #                 repulsive_force = grad_kij*alpha # For default SVGD paper, alpha = 1 (Because they dont control magnitude of repulsion) 
    #                 new_parameters[i][l] = new_parameters[i][l] + (driving_force + repulsive_force)
                    
    #                 # Adding Langevin Noise
    #                 if cfg.add_langevin_noise:
    #                     lr = optimizer.state_dict()['param_groups'][0]['lr']
    #                     kij_lav = torch.mul((kij/len(all_pgs)), torch.eye(p.data.nelement()).to(device)).to(device)
    #                     nj = torch.distributions.Normal(0, 1).sample((p.data.view(-1).shape)).to(device) # Sample noise for the flattened parameter vector
                        
    #                     langevin_noise = torch.matmul(torch.sqrt(torch.mul((2/lr),kij_lav)), nj).view_as(p.data).to(device)
    #                     new_parameters[i][l] = new_parameters[i][l] + langevin_noise
            
    #         # now average new_parameters[i][l] by number of particles
    #         for l, p in enumerate(all_pgs[i].parameters()):
    #             if p.grad is not None:
    #                 new_parameters[i][l] = new_parameters[i][l] / float(len(all_pgs))
        
    #     # take power of 2 of median    
    #     self.h_kernel = np.median(dists)**2
    #     self.h_kernel = np.sqrt(0.5 * self.h_kernel / np.log(len(all_pgs)))
        
    #     for i in range(len(all_pgs)):
    #         for l, p in enumerate(all_pgs[i].parameters()):
    #             if p.grad is not None:
    #                 p.grad.data = new_parameters[i][l]
                       
# parse command line options
parser = argparse.ArgumentParser(description="Training using Cifar10 - 3 NET")
parser.add_argument("--config", default="experiments/classifiers/adv/cifar_adv_adam_s3p_or_ens3p_R18.yml")
parser.add_argument("--resume", default=False)
args = parser.parse_args()

cfg = load_config(args.config)

trainset_cfg = cfg.dataset.train
testset_cfg = cfg.dataset.test
print(cfg)

# Setup for Wandb
wandb.init(project="ADV_CLASSIFIERS_CIFAR10", config=cfg)
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

if cfg.pretrained.allowed:
    print("=> loading pretrained model '{}'".format(cfg.pretrained.path))
    ckpt = torch.load(cfg.pretrained.path, map_location=lambda storage, loc: storage)
    ckpt['state_dict'] = {k.replace("0.", "", 1): v for k, v in ckpt['state_dict'].items()}
    net.load_state_dict(ckpt['state_dict'])
    net = net.to(device)
    
# BayesWrap - SVGD training (Particles = n, n=1/2/3.. based on config)
net = BayesWrap(net)
net = net.to(device)

# set loss
criterion = torch.nn.CrossEntropyLoss().to(device)

# set optimizers
optimizer = load_optimizer(cfg.optimizer, params=net.parameters())

# LR scheduler
# lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)
 
start_epoch = 0
best_test_clean_acc, best_test_clean_loss = 0.0, 0.0
best_test_adv_acc, best_test_adv_loss = 0.0, 0.0
best_test_clean_epoch, best_test_adv_epoch = 0, 0
if args.resume:
    print("=> loading checkpoint resuming '{}'".format(args.resume))
    ckpt = torch.load(args.resume)
    net.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch']+1
    best_test_clean_acc = ckpt['best_test_clean_acc']
    best_test_clean_loss = ckpt['best_test_clean_loss']
    best_test_adv_acc = ckpt['best_test_adv_acc']
    best_test_adv_loss = ckpt['best_test_adv_loss']

# set stylegan
# gan_path = 'models/cifar10u-cifar-ada-best-is.pkl'
# with dnnlib.util.open_url(gan_path) as f:
#     gan = legacy.load_network_pkl(f)['G_ema'].to(device)
# gan = gan.synthesis
# for p in gan.parameters():
#     p.requires_grad_(False)
# gan = move_to_device(gan, cfg, device)
# gan = gan.to(device)
# model = torch.nn.Sequential(gan, net)
# model = model.to(device)

# off-manifold attacks/AT
image_attacker = get_attack(cfg.image_attack, net)
test_attacker = PGDAttack(predict=net,
                          eps=cfg.image_attack.args.eps,
                          eps_iter=cfg.image_attack.args.eps_iter,
                          nb_iter=50,
                          clip_min=0.0,
                          clip_max=1.0)

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

# Mean, Std generated for our StyleGAN2-Ada trained on CIFAR10u-IS pretrained model.
CIFAR10_MEAN, CIFAR10_STD = (0.7456, 0.7410, 0.7232), (0.1004, 0.0990, 0.0997) # for 0 to 1 scaled images <- trained with this

def train_adv(epoch):
    progress_bar = tqdm(trainloader)

    net.train()

    image_clean_loss_meter = AverageMeter()
    image_clean_acc_meter = AverageMeter()
    image_adv_loss_meter = AverageMeter()
    image_adv_acc_meter = AverageMeter()

    for batch_idx, (images, _, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        with ctx_noparamgrad_and_eval(net):
            images_adv = image_attacker.perturb(images, labels)
        
        # now normalise the images_ladv, images_adv and images
        images = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images)
        images_adv = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images_adv)
        
        optimizer.zero_grad()
        final_img_losses_clean, final_img_losses_adv, final_img_soft_out_clean, final_img_soft_out_adv = net.get_losses_adv(images, images_adv, labels, criterion)
        final_img_losses_adv.backward() # only using adv loss for backprop
        
        if ((cfg.issvgd) and (epoch < cfg.svgd.particle_push_limit_epochs)):
            net.update_grads()
        
        optimizer.step()

        preds_clean = final_img_soft_out_clean.argmax(dim=1)
        preds_adv = final_img_soft_out_adv.argmax(dim=1)
        
        image_clean_loss_meter.update(final_img_losses_clean.item())
        image_adv_loss_meter.update(final_img_losses_adv.item())
        image_clean_acc_meter.update((preds_clean == labels).float().mean().item() * 100.0)
        image_adv_acc_meter.update((preds_adv == labels).float().mean().item() * 100.0)
        
        if batch_idx % 50 == 0:
            inputs_path = os.path.join(vis_dir, f'{epoch}_iter_{batch_idx}_inputs.png')
            adv_image_path = os.path.join(vis_dir, f'{epoch}_iter_{batch_idx}_adv_image.png')
            save_image(images[:8], inputs_path, nrow=8, padding=2, normalize=True, value_range=(-1., 1.))
            save_image(images_adv[:8], adv_image_path, nrow=8, padding=2, normalize=True, value_range=(-1., 1.))
            
        progress_bar.set_description(
            'Epoch: [{epoch}] '
            'Clean Loss: {image_clean_loss_meter.avg:.4f} '
            'Clean Acc: {image_clean_acc_meter.avg:.4f} '
            'Adv Loss: {image_adv_loss_meter.avg:.4f} '
            'Adv Acc: {image_adv_acc_meter.avg:.4f} '.format(
                epoch=epoch,
                image_clean_loss_meter=image_clean_loss_meter,
                image_clean_acc_meter=image_clean_acc_meter,
                image_adv_loss_meter=image_adv_loss_meter,
                image_adv_acc_meter=image_adv_acc_meter))
        
    return image_clean_loss_meter.avg, image_clean_acc_meter.avg, image_adv_loss_meter.avg, image_adv_acc_meter.avg

def test_adv(epoch, mode="Test"):
    
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
            
            # normalise the images, images_adv, images_ladv
            images = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images)
            images_adv = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images_adv)
            
            # Forward pass
            logits_clean = net(images)
            logits_adv = net(images_adv)
            
            # Calculate loss
            loss_clean = criterion(logits_clean, labels)
            loss_adv = criterion(logits_adv, labels)
            
            # Calculate Predictions
            pred_clean = logits_clean.argmax(dim=1)
            pred_adv = logits_adv.argmax(dim=1)
            
            # Calculate accuracy
            acc_clean = (pred_clean == labels).float().mean().item() * 100.0
            acc_adv = (pred_adv == labels).float().mean().item() * 100.0
            
        acc_clean_meter.update(acc_clean)
        acc_adv_meter.update(acc_adv)
        
        loss_clean_meter.update(loss_clean.item())
        loss_adv_meter.update(loss_adv.item())
        
        progress_bar.set_description(
            '{mode} '
            'Epoch: [{epoch}] '
            'Clean Loss: {loss_clean.val:.3f} ({loss_clean.avg:.3f}) '
            'Clean Acc: {acc_clean.val:.3f} ({acc_clean.avg:.3f}) '
            'Adv Loss: {loss_adv.val:.3f} ({loss_adv.avg:.3f}) '
            'Adv Acc: {acc_adv.val:.3f} ({acc_adv.avg:.3f}) '
            .format(
                mode=mode,
                epoch=epoch,
                loss_clean=loss_clean_meter,
                acc_clean=acc_clean_meter,
                loss_adv=loss_adv_meter,
                acc_adv=acc_adv_meter))
        
    return loss_clean_meter.avg, acc_clean_meter.avg, loss_adv_meter.avg, acc_adv_meter.avg

# ----------------------- TESTING WITH ADV - ROBUSTNESS ATTACKS on clean models -----------------------
# ADV attackers
adv_attack_budgets = [0.015, 0.02, 0.035, 0.05, 0.07, 0.1, 0.2, 0.3]
adv_attackers = []
for budget in adv_attack_budgets:
    adv_attackers.append(PGDAttack(predict=net,
                                   eps=budget,
                                   eps_iter=budget/4.0,
                                   nb_iter=50,
                                   clip_min=0.0,
                                   clip_max=1.0))
    
def robustness_test_for_ATmodels(epoch):
    
    progress_bar = tqdm(testloader)
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
                
                # normalise the images_adv
                images_adv = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images_adv)
                
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
    
# ----------------------- ROBUSTNESS ATTACKS -----------------------

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


for epoch in (range(start_epoch, cfg.num_epochs)):
    
    # Calculate distance between weights
    if (epoch == 0):
        calc_L2Dis_between_particles(epoch, log=False)
        
    # Train  
    epoch_train_clean_loss, epoch_train_clean_acc, epoch_train_adv_loss, epoch_train_adv_acc = train_adv(epoch)
    
    # Test
    epoch_test_clean_loss, epoch_test_clean_acc, epoch_test_adv_loss, epoch_test_adv_acc = test_adv(epoch)
    
    # Robustness Test
    robustness_test_for_ATmodels(epoch)
    
    # lr used
    lr = optimizer.param_groups[0]['lr']
    
    # LR scheduler
    #lr_schedule.step()
    
    # Calculate distance between particles
    calc_L2Dis_between_particles(epoch)
    
    # Log to wandb
    wandb.log({
        "epoch": epoch,
        "lr": lr,
        "train_clean_loss": epoch_train_clean_loss,
        "train_clean_acc": epoch_train_clean_acc,
        "train_adv_loss": epoch_train_adv_loss,
        "train_adv_acc": epoch_train_adv_acc,
        "test_clean_loss": epoch_test_clean_loss,
        "test_clean_acc": epoch_test_clean_acc,
        "test_adv_loss": epoch_test_adv_loss,
        "test_adv_acc": epoch_test_adv_acc
    })

    # Save best model - Clean Accuarcy Based
    if epoch_test_clean_acc > best_test_clean_acc:
        best_test_clean_acc = epoch_test_clean_acc
        best_test_clean_loss = epoch_test_clean_loss
        best_test_clean_epoch = epoch
        
        # save checkpoint
        checkpoint_path = os.path.join(output_dir, f'classifier-clean.pt')
        torch.save({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_test_clean_acc': best_test_clean_acc,
            'best_test_clean_loss': best_test_clean_loss
        }, checkpoint_path)
    
    # Save best model - Adversarial Accuracy Based
    if epoch_test_adv_acc > best_test_adv_acc:
        best_test_adv_acc = epoch_test_adv_acc
        best_test_adv_loss = epoch_test_adv_loss
        best_test_adv_epoch = epoch
        
        # save checkpoint
        checkpoint_path = os.path.join(output_dir, f'classifier-adv.pt')
        torch.save({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_test_adv_acc': best_test_adv_acc,
            'best_test_adv_loss': best_test_adv_loss
        }, checkpoint_path)
    
    # Save the trained model
    checkpoint_path = os.path.join(output_dir, f'classifier_epoch_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_test_clean_acc': best_test_clean_acc,
        'best_test_clean_loss': best_test_clean_loss,
        'best_test_adv_acc': best_test_adv_acc,
        'best_test_adv_loss': best_test_adv_loss
    }, checkpoint_path)

wandb.log({
    "best_test_clean_epoch": best_test_clean_epoch,
    "best_test_adv_epoch": best_test_adv_epoch,
    "best_test_clean_acc": best_test_clean_acc,
    "best_test_clean_loss": best_test_clean_loss,
    "best_test_adv_acc": best_test_adv_acc,
    "best_test_adv_loss": best_test_adv_loss
})

wandb.finish()