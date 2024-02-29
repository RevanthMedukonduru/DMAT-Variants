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
from koila import lazy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BayesWrap(nn.Module):
    def __init__(self, NET):
        super().__init__()

        num_particles = int(cfg.svgd.num_particles)
        self.h_kernel = 0
        self.particles = []

        for i in range(num_particles):
            self.particles.append(copy.deepcopy(NET))


        for i, particle in enumerate(self.particles):
            self.add_module(str(i), particle)

        # logging.info("num particles: %d" % len(self.particles))
        print(f"num particles: {len(self.particles)}")

    def sample_particle(self):
        return self.particles[np.random.randint(0, len(self.particles))]

    def get_particle(self, index):
        return self.particles[index]

    # This code is for calculating loss for Images, Adv Images, Latent Images : DMAT-IG.
    def get_IG_Losses(self, image, image_adv, image_ladv, labels, criterion, **kwargs):
        losses, losses_adv, losses_ladv = [], [], []
        logits, logits_adv, logits_ladv = [], [], []
        entropies, entropies_adv, entropies_ladv = [], [], []
        
        image, image_adv, image_ladv, labels = image.to(device), image_adv.to(device), image_ladv.to(device), labels.to(device)
        
        for particle in self.particles:
            l = particle(image)
            l_adv = particle(image_adv)
            l_ladv = particle(image_ladv)
            
            logits.append(l)
            logits_adv.append(l_adv)
            logits_ladv.append(l_ladv)
            prob = torch.softmax(l, 1)
            prob_adv = torch.softmax(l_adv, 1)
            prob_ladv = torch.softmax(l_ladv, 1)
            entropies.append((-prob * torch.log(prob + 1e-8)).sum(1))
            entropies_adv.append((-prob_adv * torch.log(prob_adv + 1e-8)).sum(1))
            entropies_ladv.append((-prob_ladv * torch.log(prob_ladv + 1e-8)).sum(1))
            
            loss = criterion(l, labels)
            loss_adv = criterion(l_adv, labels)
            loss_ladv = criterion(l_ladv, labels)
            
            losses.append(loss)
            losses_adv.append(loss_adv)
            losses_ladv.append(loss_ladv)
        
        final_img_losses = torch.stack(losses).mean(0)
        final_img_losses_adv = torch.stack(losses_adv).mean(0)
        final_img_losses_ladv = torch.stack(losses_ladv).mean(0)
        total_loss = (0.5 * final_img_losses_adv) + (0.5 * final_img_losses_ladv)
        
        logits = torch.stack(logits).mean(0)
        logits_adv = torch.stack(logits_adv).mean(0)
        logits_ladv = torch.stack(logits_ladv).mean(0)
        
        entropies = torch.stack(entropies).mean(0)
        entropies_adv = torch.stack(entropies_adv).mean(0)
        entropies_ladv = torch.stack(entropies_ladv).mean(0)
        
        return total_loss, final_img_losses, final_img_losses_adv, final_img_losses_ladv, logits, logits_adv, logits_ladv, entropies, entropies_adv, entropies_ladv

    
    # This code is for calculating loss for Images, Adv Images, Latent Images : DMAT Non-IG
    def get_losses(self, image, image_adv, image_ladv, labels, criterion, **kwargs):
        losses, losses_adv, losses_ladv = [], [], []
        image, image_adv, image_ladv, labels = image.to(device), image_adv.to(device), image_ladv.to(device), labels.to(device)
        
        for particle in self.particles:            
            l = particle(image)
            l_adv = particle(image_adv)
            l_ladv = particle(image_ladv)
            
            loss = criterion(l, labels)
            loss_adv = criterion(l_adv, labels)
            loss_ladv = criterion(l_ladv, labels)
            
            losses.append(loss)
            losses_adv.append(loss_adv)
            losses_ladv.append(loss_ladv)
        
        final_img_losses = torch.stack(losses).mean(0)
        final_img_losses_adv = torch.stack(losses_adv).mean(0)
        final_img_losses_ladv = torch.stack(losses_ladv).mean(0)
        total_loss = (0.5 * final_img_losses_adv) + (0.5 * final_img_losses_ladv)
        
        return total_loss, final_img_losses, final_img_losses_adv, final_img_losses_ladv
    
    def do_FP_BP_for_one_Particle(self, particle_idx, image, image_adv, image_ladv, labels, criterion):
        particle = self.particles[particle_idx]
        with torch.no_grad():
            l = particle(image)
        l_adv = particle(image_adv)
        l_ladv = particle(image_ladv)
        
        loss = criterion(l, labels)
        loss_adv = criterion(l_adv, labels)
        loss_ladv = criterion(l_ladv, labels)
        
        total_loss = (0.5 * loss_adv) + (0.5 * loss_ladv)
        
        total_loss.backward()
        torch.cuda.empty_cache()
        return total_loss, loss, loss_adv, loss_ladv
    
    def forward(self, x, **kwargs):
        logits, entropies, soft_out, stds = [], [], [], []
        return_entropy = "return_entropy" in kwargs and kwargs["return_entropy"]
        for particle in self.particles:
            x = x.to(device)
            l = particle(x)
            sft = torch.softmax(l, 1)
            soft_out.append(sft)
            logits.append(l)
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
    
    def update_grads(self):
        # print("Updating grads by: ", (1.0-float(cfg.svgd_params.p_update)))
        if np.random.rand() < (1.0-float(cfg.svgd_params.p_update)):
            return
        all_pgs = self.particles
        if self.h_kernel <= 0:
            self.h_kernel = 0.1  # 1
        dists = []
        alpha = float(cfg.svgd_params.alpha)
        new_parameters = [None] * len(all_pgs)

        for i in range(len(all_pgs)):
            new_parameters[i] = {}
            for l, p in enumerate(all_pgs[i].parameters()):
                if p.grad is None:
                    new_parameters[i][l] = None
                else:
                    new_parameters[i][l] = p.grad.data.new(
                        p.grad.data.size()).zero_()
            for j in range(len(all_pgs)):
                # if i == j:
                #     continue
                for l, params in enumerate(
                        zip(all_pgs[i].parameters(), all_pgs[j].parameters())):
                    p, p2 = params
                    if p.grad is None or p2.grad is None:
                        continue
                    if p is p2:
                        dists.append(0)
                        new_parameters[i][l] = new_parameters[i][l] + \
                            p.grad.data
                    else:
                        d = (p.data - p2.data).norm(2)
                        # if p is not p2:
                        dists.append(d.cpu().item())
                        kij = torch.exp(-(d**2) / self.h_kernel**2 / 2)
                        new_parameters[i][l] = (
                            ((new_parameters[i][l] + p2.grad.data) -
                             (d / self.h_kernel**2) * alpha) /
                            float(len(all_pgs))) * kij
        self.h_kernel = np.median(dists)
        self.h_kernel = np.sqrt(0.5 * self.h_kernel / np.log(len(all_pgs)) + 1)
        for i in range(len(all_pgs)):
            for l, p in enumerate(all_pgs[i].parameters()):
                if p.grad is not None:
                    p.grad.data = new_parameters[i][l]
    
    
    def freeze_backbone(self, exclude=[], include=[]):
        if include and len(include) > 0:
            for name, param in self.named_parameters():
                # if keyword in include is available in name, freeze it
                if any([e in name for e in include]):
                    param.requires_grad = False
        else:
            for particle in self.particles:    
                for name, param in particle.named_parameters():
                    # any string in name is there, then dont freeze it
                    if not any([e in name for e in exclude]):
                        param.requires_grad = False

# parse command line options
parser = argparse.ArgumentParser(description="On-manifold adv training")
parser.add_argument("--config", default="our_experiments/classifiers/img_manifold_pgd5_sgd_svgd3p.yml")
parser.add_argument("--resume", default="")
args = parser.parse_args()

cfg = load_config(args.config)
trainset_cfg = cfg.dataset.train
testset_cfg = cfg.dataset.test
print(cfg)

IS_ENSnP_or_SVGD1P_TRAINING = cfg.IS_ENSnP_or_SVGD1P_TRAINING
IS_IG_TRAINING = cfg.IS_IG_TRAINING
print(f"IS_ENSnP_or_SVGD1P_TRAINING: {IS_ENSnP_or_SVGD1P_TRAINING}", f"IS_IG_TRAINING: {IS_IG_TRAINING}")
print(f"DTYPE of IS_ENSnP_or_SVGD1P_TRAINING: {type(IS_ENSnP_or_SVGD1P_TRAINING)}", f"DTYPE of IS_IG_TRAINING: {type(IS_IG_TRAINING)}")

# Setup for Wandb
wandb.init(project="ATTACKS_RS50_DMAT_BUDG", config=cfg)
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
set_device(cfg)
set_random_seed(cfg)
cudnn.benchmark = True

# set classifier
net = get_classifier(cfg, cfg.classifier)
net = net.to(device)
print(f"INITIAL TEST: ", len(list(net.parameters())))

net = BayesWrap(net)
net = net.to(device)
# Visualization(net).structure_graph()
print(f"INITIAL TEST 2: ", len(list(net.parameters())))

# set optimizers
optimizer = load_optimizer(cfg.optimizer, params=[p for p in net.parameters() if p.requires_grad])

if cfg.scheduler.type == 'cyclic':
    lr_schedule = lambda t: np.interp([t], cfg.scheduler.args.lr_epochs, cfg.scheduler.args.lr_values)[0]
else:
    lr_schedule = None

start_epoch = 0
if args.resume:
    print("=> loading checkpoint '{}'".format(args.resume))
    ckpt = torch.load(args.resume)
    net.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch']
    
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

image_attacker = get_attack(cfg.image_attack, net)
latent_attacker = get_attack(cfg.latent_attack, model)

test_attacker = PGDAttack(predict=net,
                          eps=cfg.image_attack.args.eps,
                          eps_iter=cfg.image_attack.args.eps_iter,
                          nb_iter=50,
                          clip_min=-1.0,
                          clip_max=1.0)

test_latent_attacker = PGDAttack(predict=model,
                                 eps=cfg.latent_attack.args.eps,
                                 eps_iter=cfg.latent_attack.args.eps_iter,
                                 nb_iter=50, 
                                 clip_max=None, clip_min=None)

# set dataset, dataloader
dataset = get_dataset(cfg)
transform = get_transform(cfg)
trainset = dataset(root=trainset_cfg.path, train=True) #FIXME0
testset = dataset(root=testset_cfg.path, train=False) #FIXME0 - Is transforms required? (transform=transform.classifier_testing)
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


def train_IG(epoch):
    progress_bar = tqdm(trainloader)

    net.train()
    gan.eval()

    image_loss_meter = AverageMeter()
    image_adv_loss_meter = AverageMeter()
    image_ladv_loss_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    
    infogain_meter_adv = AverageMeter()
    infogain_meter_ladv = AverageMeter()
    total_infogain_meter = AverageMeter()
    
    overall_loss_meter = AverageMeter()

    for batch_idx, (images, latents, labels) in enumerate(progress_bar):
        images, latents, labels = images.to(device), latents.to(device), labels.to(device)

        lr = cfg.optimizer.args.lr
        if lr_schedule is not None:
            lr = lr_schedule(epoch + (batch_idx + 1) / len(trainloader))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        with ctx_noparamgrad_and_eval(model):
            images_adv = image_attacker.perturb(images, labels)
            latents_adv = latent_attacker.perturb(latents, labels)

        images_ladv = gan(latents_adv).detach() 
        
        # Do I need to clamp values which are not in the range of -1 to 1?
        # images_ladv = transform.classifier_preprocess_layer(images_ladv) #FIXME1

        # Do I need to normalise? #FIXME2
        # images = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images)
        # images_adv = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images_adv)
        # images_ladv = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images_ladv)
           
        optimizer.zero_grad()
        
        # The entropies we will recieve from get_losses func is (Ind entropies stacked, then meaned over particles)
        total_loss, img_loss, img_loss_adv, img_loss_ladv, logits, logits_adv, logits_ladv, entropies, entropies_adv, entropies_ladv = net.get_IG_Losses(images, images_adv, images_ladv, labels, criterion)
        
        # Calculate InfoGain
        img_prob = torch.softmax(logits, 1)
        img_prob_adv = torch.softmax(logits_adv, 1)
        img_prob_ladv = torch.softmax(logits_ladv, 1)
        
        # The entropies we calculated below are (mean of logits, then softmax, then entropies)
        img_entropies = (-img_prob * torch.log(img_prob + 1e-8)).sum(1)
        img_entropies_adv = (-img_prob_adv * torch.log(img_prob_adv + 1e-8)).sum(1)
        img_entropies_ladv = (-img_prob_ladv * torch.log(img_prob_ladv + 1e-8)).sum(1)
        
        # Loss IG
        ig_loss_img = img_entropies - entropies
        ig_loss_img_adv = img_entropies_adv - entropies_adv
        ig_loss_img_ladv = img_entropies_ladv - entropies_ladv
        
        final_ig_loss_adv = torch.abs(ig_loss_img_adv - ig_loss_img).mean(0) # Adversarial - Clean
        final_ig_loss_ladv = torch.abs(ig_loss_img_ladv - ig_loss_img).mean(0) # Latent Adversarial - Clean
        total_ig_loss = (0.5 * final_ig_loss_adv) + (0.5 * final_ig_loss_ladv) # Total IG Loss = 50% Adversarial IG Loss + 50% Latent Adversarial IG Loss
        
        # get the hyper-parameter to control IG
        lambda1 = cfg.ig_params.ig_combination_ratio.lambda1
        
        # Combining with (lambda1) proportion of total loss (50% image loss + 50% latent loss = total loss) and (lambda2) proportion of IG loss
        overall_loss = total_loss + (lambda1 * total_ig_loss)
            
        overall_loss.backward()
        net.update_grads()
        optimizer.step()

        image_loss_meter.update(img_loss.item()) # Image Loss for Clean Images
        image_adv_loss_meter.update(img_loss_adv.item()) # Image Loss for Adversarial Images
        image_ladv_loss_meter.update(img_loss_ladv.item()) # Image Loss for Latent Vector based Adversarial Images
        ce_loss_meter.update(total_loss.item()) # Cross Entropy Loss (CE) = 50% Adversarial Image Loss + 50% Latent Vector based Adversarial Image Loss
        
        infogain_meter_adv.update(final_ig_loss_adv.item()) # IG Loss for Image based Adversarial Images
        infogain_meter_ladv.update(final_ig_loss_ladv.item()) # IG Loss for Latent Vector based Adversarial Images
        total_infogain_meter.update(total_ig_loss.item()) # Total IG Loss
        
        overall_loss_meter.update(overall_loss.item()) # Overall Loss (CE + x.IG)

        if batch_idx % 50 == 0:
            inputs_path = os.path.join(vis_dir, f'{epoch}_iter_{batch_idx}_inputs.png')
            adv_image_path = os.path.join(vis_dir, f'{epoch}_iter_{batch_idx}_adv_image.png')
            adv_latent_path = os.path.join(vis_dir, f'{epoch}_iter_{batch_idx}_adv_latent.png')
            save_image(images[:8], inputs_path, nrow=8, padding=2, normalize=True, value_range=(0., 1.))
            save_image(images_adv[:8], adv_image_path, nrow=8, padding=2, normalize=True, value_range=(0., 1.))
            save_image(images_ladv[:8], adv_latent_path, nrow=8, padding=2, normalize=True, value_range=(0., 1.))

        progress_bar.set_description(
            'E: [{epoch}] '
            'Img Lo: {image_loss.val:.3f} ({image_loss.avg:.3f}) '
            'Adv Lo: {image_loss_adv.val:.3f} ({image_loss_adv.avg:.3f}) '
            'LAdv Lo: {image_loss_ladv.val:.3f} ({image_loss_ladv.avg:.3f}) '
            'CE Loss: {ce_loss.val:.3f} ({ce_loss.avg:.3f})'
            'AIG Lo: {ig_loss_adv.val:.3f} ({ig_loss_adv.avg:.3f}) '
            'LIG Lo: {ig_loss_ladv.val:.3f} ({ig_loss_ladv.avg:.3f}) '
            'T-IG Loss: {total_ig_loss.val:.3f} ({total_ig_loss.avg:.3f}) '
            'Ovall Loss: {overall_loss.val:.3f} ({overall_loss.avg:.3f}) '.format(
                epoch=epoch,
                image_loss=image_loss_meter,
                image_loss_adv=image_adv_loss_meter,
                image_loss_ladv=image_ladv_loss_meter,
                ce_loss=ce_loss_meter,
                ig_loss_adv=infogain_meter_adv,
                ig_loss_ladv=infogain_meter_ladv,
                total_ig_loss=total_infogain_meter,
                overall_loss=overall_loss_meter))
        
    return ce_loss_meter.avg, infogain_meter_adv.avg, infogain_meter_ladv.avg, total_infogain_meter.avg, overall_loss_meter.avg

def train(epoch):
    
    progress_bar = tqdm(trainloader)

    net.train()
    gan.eval()

    image_loss_meter = AverageMeter()
    image_adv_loss_meter = AverageMeter()
    image_ladv_loss_meter = AverageMeter()
    overall_loss_meter = AverageMeter()

    for batch_idx, (images, latents, labels) in enumerate(progress_bar):
        images, latents, labels = images.to(device), latents.to(device), labels.to(device)
        
        lr = cfg.optimizer.args.lr
        if lr_schedule is not None:
            lr = lr_schedule(epoch + (batch_idx + 1) / len(trainloader))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        with ctx_noparamgrad_and_eval(net):
            images_adv = image_attacker.perturb(images, labels)
        
        with ctx_noparamgrad_and_eval(model):
            latents_adv = latent_attacker.perturb(latents, labels)
        
        with ctx_noparamgrad_and_eval(gan):
            images_ladv = gan(latents_adv).detach()
        
        torch.cuda.empty_cache()
        
        # FP, BP
        optimizer.zero_grad()
        
        """
        # Removed Old logic of SVGD, updated with Optimal SVGD computation - Memory Efficient
        total_loss, img_loss, img_loss_adv, img_loss_ladv = net.get_losses(images, images_adv, images_ladv, labels, criterion)
        total_loss.backward()
        """
        
        total_losses, losses, losses_adv, losses_ladv = [], [], [], []
        for particle_id in range(int(cfg.svgd.num_particles)):
            total_loss_per_par, img_loss_per_par, img_loss_adv_per_par, img_loss_ladv_per_par = net.do_FP_BP_for_one_Particle(particle_id, images, images_adv, images_ladv, labels, criterion)
            losses.append(img_loss_per_par)
            losses_adv.append(img_loss_adv_per_par)
            losses_ladv.append(img_loss_ladv_per_par)
            total_losses.append(total_loss_per_par)
        total_loss = torch.stack(total_losses).mean(0)
        img_loss = torch.stack(losses).mean(0)
        img_loss_adv = torch.stack(losses_adv).mean(0)
        img_loss_ladv = torch.stack(losses_ladv).mean(0)
        
        if not IS_ENSnP_or_SVGD1P_TRAINING:
            net.update_grads() # PUSHING SVGD PARTICLES if its not ensembling and not just 1 particle.
        optimizer.step()

        image_loss_meter.update(img_loss.item()) # Image Loss for Clean Images
        image_adv_loss_meter.update(img_loss_adv.item()) # Image Loss for Adversarial Images
        image_ladv_loss_meter.update(img_loss_ladv.item()) # Image Loss for Latent Vector based Adversarial Images
        overall_loss_meter.update(total_loss.item()) # Overall Loss (CE only)
        
        if batch_idx % 50 == 0:
            inputs_path = os.path.join(vis_dir, f'{epoch}_iter_{batch_idx}_inputs.png')
            adv_image_path = os.path.join(vis_dir, f'{epoch}_iter_{batch_idx}_adv_image.png')
            adv_latent_path = os.path.join(vis_dir, f'{epoch}_iter_{batch_idx}_adv_latent.png')
            save_image(images[:8], inputs_path, nrow=8, padding=2, normalize=True, value_range=(0., 1.))
            save_image(images_adv[:8], adv_image_path, nrow=8, padding=2, normalize=True, value_range=(0., 1.))
            save_image(images_ladv[:8], adv_latent_path, nrow=8, padding=2, normalize=True, value_range=(0., 1.))

        progress_bar.set_description(
            'TRAIN E: [{epoch}] '
            'LR: {lr:.6f} '
            'Img Lo: {image_loss.val:.3f} ({image_loss.avg:.3f}) '
            'Adv Lo: {image_loss_adv.val:.3f} ({image_loss_adv.avg:.3f}) '
            'LAdv Lo: {image_loss_ladv.val:.3f} ({image_loss_ladv.avg:.3f}) '
            'Ovl Lo: {overall_loss.val:.3f} ({overall_loss.avg:.3f}) '.format(
                epoch=epoch,
                lr = lr,
                image_loss=image_loss_meter,
                image_loss_adv=image_adv_loss_meter,
                image_loss_ladv=image_ladv_loss_meter,
                overall_loss=overall_loss_meter))
        
        torch.cuda.empty_cache()
        
    return overall_loss_meter.avg

def test(epoch, mode="Test"):
    progress_bar = tqdm(testloader)
    net.eval()
    gan.eval()

    # Clean Image
    loss_clean_meter = AverageMeter()
    acc_clean_meter = AverageMeter()
    
    # Adversarial Image - Image attack
    loss_adv_meter = AverageMeter()
    acc_adv_meter = AverageMeter()
    
    # Latent Vector based Adversarial Image - Latent attack
    loss_ladv_meter = AverageMeter()
    acc_ladv_meter = AverageMeter()

    for batch_idx, (images, latents, labels) in enumerate(progress_bar):
        images, latents, labels = images.to(device), latents.to(device), labels.to(device)
            
        with ctx_noparamgrad_and_eval(model):
            latents_adv = test_latent_attacker.perturb(latents, labels)       
        
        with ctx_noparamgrad_and_eval(gan):
            images_ladv = gan(latents_adv).detach()
        
        with ctx_noparamgrad_and_eval(net):
            images_adv = test_attacker.perturb(images, labels)    
            
            # here we need to preprocess the perturbed latent Vector based Adversarial Images as well before passing to the classifier
            # images_ladv = transform.classifier_preprocess_layer(images_ladv) # input -> Clamps to [-1, 1], scales to [0, 1] -> returned
            
            # normalise the images, images_adv, images_ladv
            # images = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images)
            # images_adv = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images_adv)
            # images_ladv = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images_ladv)
            
        # Forward pass
        logits_clean = net(images)
        logits_adv = net(images_adv)
        logits_ladv = net(images_ladv)
        
        # Calculate loss
        loss_clean = criterion(logits_clean, labels)
        loss_adv = criterion(logits_adv, labels)
        loss_ladv = criterion(logits_ladv, labels)
        
        # Calculate Predictions
        pred_clean = logits_clean.argmax(dim=1)
        pred_adv = logits_adv.argmax(dim=1)
        pred_ladv = logits_ladv.argmax(dim=1)
        
        # Calculate accuracy
        acc_clean = (pred_clean == labels).float().mean().item() * 100.0
        acc_adv = (pred_adv == labels).float().mean().item() * 100.0
        acc_ladv = (pred_ladv == labels).float().mean().item() * 100.0
            
        acc_clean_meter.update(acc_clean)
        acc_adv_meter.update(acc_adv)
        acc_ladv_meter.update(acc_ladv)
        
        loss_clean_meter.update(loss_clean.item())
        loss_adv_meter.update(loss_adv.item())
        loss_ladv_meter.update(loss_ladv.item())
    
        progress_bar.set_description(
            'Ep: [{epoch}] '
            'Cl Lo: ({loss_clean.avg:.3f}) '
            'Cl Ac:  ({acc_clean.avg:.3f}) '
            'Ad Lo: ({loss_adv.avg:.3f}) '
            'Ad Ac: ({acc_adv.avg:.3f}) '
            'LA Lo: ({loss_ladv.avg:.3f}) '
            'LA Ac: ({acc_ladv.avg:.3f}) '.format(epoch=epoch, loss_clean=loss_clean_meter, acc_clean=acc_clean_meter, loss_adv=loss_adv_meter, acc_adv=acc_adv_meter, loss_ladv=loss_ladv_meter, acc_ladv=acc_ladv_meter))

    return loss_clean_meter.avg, acc_clean_meter.avg, loss_adv_meter.avg, acc_adv_meter.avg, loss_ladv_meter.avg, acc_ladv_meter.avg

best_test_clean_acc, best_test_clean_loss = 0.0, 0.0
best_test_adv_acc, best_test_adv_loss = 0.0, 0.0
best_test_ladv_acc, best_test_ladv_loss = 0.0, 0.0

for epoch in range(start_epoch, cfg.num_epochs):
    if cfg.distributed:
        train_sampler.set_epoch(epoch)
    
    # Train
    if not IS_IG_TRAINING:
        train_overall_loss = train(epoch)
    else:
        train_ce_loss, train_ig_loss_adv, train_ig_loss_ladv, train_total_ig_loss, train_overall_loss = train_IG(epoch)
    
    # Test on test data
    test_clean_loss, test_clean_acc, test_adv_loss, test_adv_acc, test_ladv_loss, test_ladv_acc = test(epoch)
    
    # lr used
    lr = optimizer.param_groups[0]['lr']
    
    # Save best model - Clean Accuarcy Based
    if test_clean_acc > best_test_clean_acc:
        best_test_clean_acc = test_clean_acc
        best_test_clean_loss = test_clean_loss
        
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
    if test_adv_acc > best_test_adv_acc:
        best_test_adv_acc = test_adv_acc
        best_test_adv_loss = test_adv_loss
        
        # save checkpoint
        checkpoint_path = os.path.join(output_dir, f'classifier-adv.pt')
        torch.save({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_test_adv_acc': best_test_adv_acc,
            'best_test_adv_loss': best_test_adv_loss
        }, checkpoint_path)
    
    # Save best model - Latent Adversarial Accuracy Based
    if test_ladv_acc > best_test_ladv_acc:
        best_test_ladv_acc = test_ladv_acc
        best_test_ladv_loss = test_ladv_loss
        
        # save checkpoint
        checkpoint_path = os.path.join(output_dir, f'classifier-ladv.pt')
        torch.save({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_test_ladv_acc': best_test_ladv_acc,
            'best_test_ladv_loss': best_test_ladv_loss
        }, checkpoint_path)
    
    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, f'classifier-last.pt')
    torch.save({
        'epoch': epoch + 1,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_test_clean_acc': best_test_clean_acc,
        'best_test_clean_loss': best_test_clean_loss,
        'best_test_adv_acc': best_test_adv_acc,
        'best_test_adv_loss': best_test_adv_loss,
        'best_test_ladv_acc': best_test_ladv_acc,
        'best_test_ladv_loss': best_test_ladv_loss
    }, checkpoint_path)
            
    # Log results
    logging.info(
        "Epoch: [{epoch}] "
        "lr: {lr} "
        "train_overall_loss: {train_overall_loss:.3f}"
        "test_clean_loss: {test_clean_loss:.3f} "
        "test_clean_acc: {test_clean_acc:.3f} "
        "test_adv_loss: {test_adv_loss:.3f} "
        "test_adv_acc: {test_adv_acc:.3f} ".format(
            epoch=epoch,
            lr=lr,
            train_overall_loss=train_overall_loss,
            test_clean_loss=test_clean_loss,
            test_clean_acc=test_clean_acc,
            test_adv_loss=test_adv_loss,
            test_adv_acc=test_adv_acc,
        )
    )

    # Log results to wandb
    if IS_IG_TRAINING:
        wandb.log({
            "epoch": epoch,
            "lr": lr,
            "train_ce_loss": train_ce_loss,
            "train_ig_loss_adv": train_ig_loss_adv,
            "train_ig_loss_ladv": train_ig_loss_ladv,
            "train_total_ig_loss": train_total_ig_loss,
            "train_overall_loss": train_overall_loss,
            "test_clean_loss": test_clean_loss,
            "test_clean_acc": test_clean_acc,
            "test_adv_loss": test_adv_loss,
            "test_adv_acc": test_adv_acc,
            "test_ladv_loss": test_ladv_loss,
            "test_ladv_acc": test_ladv_acc
        })
    else:     
        wandb.log({
            "epoch": epoch,
            "lr": lr,
            "train_overall_loss": train_overall_loss,
            "test_clean_loss": test_clean_loss,
            "test_clean_acc": test_clean_acc,
            "test_adv_loss": test_adv_loss,
            "test_adv_acc": test_adv_acc,
            "test_ladv_loss": test_ladv_loss,
            "test_ladv_acc": test_ladv_acc
        })

wandb.log({
    "best_test_clean_acc": best_test_clean_acc,
    "best_test_clean_loss": best_test_clean_loss,
    "best_test_adv_acc": best_test_adv_acc,
    "best_test_adv_loss": best_test_adv_loss,
    "best_test_ladv_acc": best_test_ladv_acc,
    "best_test_ladv_loss": best_test_ladv_loss
})
    
wandb.finish()
