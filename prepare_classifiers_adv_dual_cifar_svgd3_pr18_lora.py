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

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from opendelta import LoraModel
from bigmodelvis import Visualization
from model import count_trainable_parameters as count_trainable_parameters_lora

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SVGD_LORA_PARTICLES_OPTIMAL(nn.Module):
    def __init__(self, model):
        super().__init__()
        
        self.back_bone = model.to(device)
        self.num_particles = int(cfg.svgd.num_particles)
        
        self.delta_models = []
        self.h_kernel = 0
  
        # freeze the backbone model
        self.freeze_backbone()
        
        for i in range(self.num_particles):
            print(f"Creating Lora Model: {i+1}")
            self.delta_models.append(LoraModel(backbone_model=model, modified_modules=['linear'], lora_r=16))
            self.delta_models[i].detach()
        
        self.attach_all_lora()
        print("After Creating and Attaching all lora models, frozen backbone model is -->")
        # Visualization(self.back_bone).structure_graph() # Before detaching all lora models
        trainable_params, delta_params, total_params = count_trainable_parameters_lora(self.back_bone)
        print(f"\nIn Lora Model with {self.num_particles} particles is: Trainable Params: {trainable_params}, Delta Params: {delta_params}, Total Params: {total_params}")
        self.detach_all_lora()
        
        print("After Creating and Detaching all lora models, frozen backbone model is -->")
        # Visualization(self.back_bone).structure_graph() # After detaching all lora models
        trainable_params, delta_params, total_params = count_trainable_parameters_lora(self.back_bone)
        print(f"\nIn Lora Model with {self.num_particles} particles is: Trainable Params: {trainable_params}, Delta Params: {delta_params}, Total Params: {total_params}")
        
        #  check if lora is frozen
        print(f"\n Checking if model is frozen or not for {self.num_particles} particles -->")
        self.attach_all_lora()
        for name, param in self.back_bone.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.requires_grad)
        self.detach_all_lora()

    def print_grad_true_params(self):
        for name, param in self.back_bone.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
                    
    def get_particle(self, index):
        self.delta_models[index].attach()
        return self.delta_models[index]
    
    def detach_lora_particle(self, index):
        self.delta_models[index].detach()
    
    def forward(self, image, **kwargs):
        self.detach_all_lora() # detach all lora models if attached
        logits, entropies, soft_out, stds = [], [], [], []
        return_entropy = "return_entropy" in kwargs and kwargs["return_entropy"]
        image = image.to(device)
        for i in range(self.num_particles):
            self.delta_models[i].attach() # attach the lora model
            l = self.back_bone(image)
            sft = torch.softmax(l, 1)
            soft_out.append(sft)
            logits.append(l)
            if return_entropy:
                l = torch.softmax(l, 1)
                entropies.append((-l * torch.log(l + 1e-8)).sum(1))
            self.delta_models[i].detach() # detach the lora model
        # Average the logits, soft_out and entropies
        logits = torch.stack(logits).mean(0)
        stds = torch.stack(soft_out).std(0)
        soft_out = torch.stack(soft_out).mean(0)
        if return_entropy:
            entropies = torch.stack(entropies).mean(0)
            return logits, entropies, soft_out, stds
        return logits
    
    # This code is for calculating loss for Images, Adv Images, Latent Images : DMAT-IG.
    """
    def get_IG_Losses(self, image, image_adv, image_ladv, labels, criterion, **kwargs):
        self.detach_all_lora() # detach all lora models if attached
        losses, losses_adv, losses_ladv = [], [], []
        logits, logits_adv, logits_ladv = [], [], []
        entropies, entropies_adv, entropies_ladv = [], [], []
        
        image, image_adv, image_ladv, labels = image.to(device), image_adv.to(device), image_ladv.to(device), labels.to(device)
        
        for i in range(self.num_particles):
            self.delta_models[i].attach()
            
            l = self.back_bone(image)
            l_adv = self.back_bone(image_adv)
            l_ladv = self.back_bone(image_ladv)
            
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
            
            self.delta_models[i].detach()
            
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
    """
    
    # This code is for calculating loss for Images, Adv Images, Latent Images : DMAT Non-IG
    def get_losses(self, image, image_adv, image_ladv, labels, criterion, **kwargs):
        self.detach_all_lora() # detach all lora models if attached
        losses, losses_adv, losses_ladv = [], [], []
        image, image_adv, image_ladv, labels = image.to(device), image_adv.to(device), image_ladv.to(device), labels.to(device)
        for i in range(self.num_particles):
            self.delta_models[i].attach()
            
            l = self.back_bone(image)
            l_adv = self.back_bone(image_adv)
            l_ladv = self.back_bone(image_ladv)
            
            loss = criterion(l, labels)
            loss_adv = criterion(l_adv, labels)
            loss_ladv = criterion(l_ladv, labels)
            
            losses.append(loss)
            losses_adv.append(loss_adv)
            losses_ladv.append(loss_ladv)
            
            self.delta_models[i].detach()
        
        final_img_losses = torch.stack(losses).mean(0)
        final_img_losses_adv = torch.stack(losses_adv).mean(0)
        final_img_losses_ladv = torch.stack(losses_ladv).mean(0)
        total_loss = (0.5 * final_img_losses_adv) + (0.5 * final_img_losses_ladv)
        
        return total_loss, final_img_losses, final_img_losses_adv, final_img_losses_ladv
        
    # This code is for calculating loss for Images, Adv Images - Adv Training (AT).
    """
    def get_losses(self, image, image_l, labels, criterion, **kwargs):
        self.detach_all_lora() # detach all lora models if attached
        img_losses, lat_losses = [], []
        for i in range(self.num_particles):
            image, image_l, labels = image.to(device), image_l.to(device), labels.to(device)
            self.delta_models[i].attach()
            l = self.delta_models[i].backbone_model(image)
            l1 = self.delta_models[i].backbone_model(image_l)
            img_loss = criterion(l, labels)
            lat_loss = criterion(l1, labels)
            img_losses.append(img_loss)
            lat_losses.append(lat_loss)
            self.delta_models[i].detach()
        final_img_losses = torch.stack(img_losses).mean(0)
        final_lat_losses = torch.stack(lat_losses).mean(0)
        total_loss = (0.5 * final_img_losses) + (0.5 * final_lat_losses)
        return total_loss, final_img_losses, final_lat_losses
    """
    
    # This code is for calculating loss for Images - General Training.
    """
    def get_losses(self, image, labels, criterion, **kwargs): # without latent image
        losses = []
        for i in range(self.num_particles):
            self.delta_models[i].attach()
            l = self.delta_models[i].backbone_model(image)
            loss = criterion(l, labels)
            losses.append(loss)
            self.delta_models[i].detach()
        losses = torch.stack(losses).mean(0)
        return losses
    """

    def update_grads(self):
        if np.random.rand() < 0.95:
            return
        
        all_pgs = self.delta_models
        if self.h_kernel <= 0:
            self.h_kernel = 0.001  # 1
        dists = []
        alpha = 0.001  # NEED TO TEST either [0.001, 0.005, 0.01]
        new_parameters = [None] * len(all_pgs)

        print("UPDATE GRADS FUNCTION: ALREADY ENABLED GRADS")
        self.print_grad_true_params()
        
        for i in range(len(all_pgs)):
            new_parameters[i] = {}
            for l, p in enumerate(all_pgs[i].parameters()):
                if p.grad is None:
                    new_parameters[i][l] = None
                else:
                    new_parameters[i][l] = p.grad.data.new(
                        p.grad.data.size()).zero_()
                    # print(f"particle: {i} has grad: {p.grad.data.size()}")
                    
            for j in range(len(all_pgs)):
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

    
    def freeze_backbone(self):
        for param in self.back_bone.parameters():
            param.requires_grad = False
            
    def detach_all_lora(self):
        for i in range(self.num_particles):
            self.delta_models[i].detach()
    
    def attach_all_lora(self):
        self.detach_all_lora()
        for i in range(self.num_particles):
            try:
                self.delta_models[i].attach()
            except Exception as e:
                print(e)              


# parse command line options
parser = argparse.ArgumentParser(description="On-manifold adv training using Cifar10 - SVGD3 Particle")
parser.add_argument("--config", default="experiments/classifiers/cifar_manifold_pgd5_sgd_par3_pr18_lora.yml") #ONRUN_VERIFY
parser.add_argument("--resume", default=False)
args = parser.parse_args()

cfg = load_config(args.config)

# Setup for Wandb
wandb.init(project="ATTACKS_PRERES18_IS_PRETRAINED", config=cfg)
logging.info(cfg)

trainset_cfg = cfg.dataset.train
testset_cfg = cfg.dataset.test
print(cfg)

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

net = SVGD_LORA_PARTICLES_OPTIMAL(net)
net = net.to(device)

# save intial checkpoint
checkpoint_path = os.path.join(output_dir, f'classifier-initial.pt')
torch.save({
    'state_dict': [net.delta_models[i].state_dict() for i in range(int(cfg.svgd.num_particles))],
}, checkpoint_path)

# set loss
criterion = torch.nn.CrossEntropyLoss().to(device)

# set optimizers
net.attach_all_lora()
for name, param in net.back_bone.named_parameters():
    if param.requires_grad:
        print(name, param.shape, param.requires_grad)
    
optimizer = load_optimizer(cfg.optimizer, params=[p for p in net.back_bone.parameters() if p.requires_grad])
net.detach_all_lora()

# LR scheduler
lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

start_epoch = 0
if args.resume:
    print("=> loading checkpoint '{}'".format(args.resume))
    ckpt = torch.load(args.resume)
    net.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch']

# set stylegan
gan_path = 'models/cifar10u-cifar-ada-best-is.pkl'
with dnnlib.util.open_url(gan_path) as f:
    gan = legacy.load_network_pkl(f)['G_ema'].to(device)

gan = gan.synthesis
net = net.to(device)

for p in gan.parameters():
    p.requires_grad_(False)
gan = move_to_device(gan, cfg, device)
model = torch.nn.Sequential(gan, net)
model = model.to(device)

image_attacker = get_attack(cfg.image_attack, net)
latent_attacker = get_attack(cfg.latent_attack, model)

test_attacker = PGDAttack(predict=net,
                          eps=cfg.image_attack.args.eps,
                          eps_iter=cfg.image_attack.args.eps_iter,
                          nb_iter=50,
                          clip_min=0.0,
                          clip_max=1.0)

test_latent_attacker = PGDAttack(predict=model,
                                 eps=cfg.latent_attack.args.eps,
                                 eps_iter=cfg.latent_attack.args.eps_iter,
                                 nb_iter=50, 
                                 clip_max=None, 
                                 clip_min=None)


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

def train_IG(epoch):
    progress_bar = tqdm(trainloader)

    net.train()
    gan.eval()

    image_loss_meter = AverageMeter()
    image_adv_loss_meter = AverageMeter()
    image_ladv_loss_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    overall_loss_meter = AverageMeter()
    infogain_meter_adv = AverageMeter()
    infogain_meter_ladv = AverageMeter()
    total_infogain_meter = AverageMeter()

    kwargs = {"return_entropy": True}
    
    for batch_idx, (images, latents, labels) in enumerate(progress_bar):
        images, latents, labels = images.to(device), latents.to(device), labels.to(device)
        
        with ctx_noparamgrad_and_eval(model):
            images_adv = image_attacker.perturb(images, labels) # For this already the image is scaled to [0, 1]
            latents_adv = latent_attacker.perturb(latents, labels)

        images_ladv = gan(latents_adv).detach()
        images_ladv = transform.classifier_preprocess_layer(images_ladv)
        
        # now normalise the images_ladv, images_adv and images
        images = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images)
        images_adv = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images_adv)
        images_ladv = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images_ladv)
        
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
        # final_ig_loss = torch.abs(((0.5*ig_loss_img_adv) + (0.5*ig_loss_img_ladv)) - ig_loss_img).mean(0) # TO BE FIXED
        
        final_ig_loss_adv = torch.abs(ig_loss_img_adv - ig_loss_img).mean(0) # Adversarial - Clean
        final_ig_loss_ladv = torch.abs(ig_loss_img_ladv - ig_loss_img).mean(0) # Latent Adversarial - Clean
        total_ig_loss = (0.5 * final_ig_loss_adv) + (0.5 * final_ig_loss_ladv) # Total IG Loss = 50% Adversarial IG Loss + 50% Latent Adversarial IG Loss
        
        # get the hyper-parameter to control IG
        lambda1 = cfg.ig_combination_ratio.lambda1
        
        # Combining with (lambda1) proportion of total loss (50% image loss + 50% latent loss = total loss) and (lambda2) proportion of IG loss
        overall_loss = total_loss + (lambda1 * total_ig_loss)
        
        net.attach_all_lora()
        overall_loss.backward()
        net.update_grads()
        optimizer.step()
        net.detach_all_lora()

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
            save_image(images[:8], inputs_path, nrow=8, padding=2, normalize=True, value_range=(0.0, 1.)) # ONRUN_VERIFY
            save_image(images_adv[:8], adv_image_path, nrow=8, padding=2, normalize=True, value_range=(0.0, 1.))
            save_image(images_ladv[:8], adv_latent_path, nrow=8, padding=2, normalize=True, value_range=(0.0, 1.))

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
        
        with ctx_noparamgrad_and_eval(model):
            images_adv = image_attacker.perturb(images, labels) # For this already the image is scaled to [0, 1]
            latents_adv = latent_attacker.perturb(latents, labels)

        images_ladv = gan(latents_adv).detach()
        images_ladv = transform.classifier_preprocess_layer(images_ladv)
        
        # now normalise the images_ladv, images_adv and images
        images = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images)
        images_adv = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images_adv)
        images_ladv = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images_ladv)
        
        optimizer.zero_grad()
        
        # The entropies we will recieve from get_losses func is (Ind entropies stacked, then meaned over particles)
        overall_loss, img_loss, img_loss_adv, img_loss_ladv = net.get_losses(images, images_adv, images_ladv, labels, criterion)
        
        # see what overall_loss is returning
        print("Overall Loss: ")
        print(overall_loss)
                
        net.attach_all_lora()
        #print state dict
        print("State Dict: ")
        Visualization(net.back_bone).structure_graph()
        print(net.back_bone.state_dict().keys())
        print("Named Parameters: ")
        for name, param in net.back_bone.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.requires_grad)
        print("Parameters: ")
        for param in net.back_bone.parameters():
            if param.requires_grad:
                print(param.shape, param.requires_grad)
        
        overall_loss.backward()
        
        # get the gradients for all particles
        # prev_grads = [p.grad.clone() for p in net.back_bone.parameters() if p.requires_grad and p.grad is not None]
        
        # get grad particle by particle
        # print("There are net.num_particles: ", net.num_particles)
        # prev_grads = []
        # for i in range(net.num_particles):
        #     net.delta_models[i].attach()
        #     for p in net.delta_models[i].backbone_model.parameters():
        #         if p.requires_grad and p.grad is not None:
        #             prev_grads.append(p.grad.clone())
        #     net.delta_models[i].detach()
        
        net.update_grads()
        
        # get the gradients for all particles
        # cur_grads = [p.grad.clone() for p in net.back_bone.parameters() if p.requires_grad and p.grad is not None]
        
        optimizer.step()
        net.detach_all_lora()
        
        # if not all(torch.equal(prev_grads[i], cur_grads[i]) for i in range(len(prev_grads))):
        #     print("Gradients are Updated.")
            
        image_loss_meter.update(img_loss.item()) # Image Loss for Clean Images
        image_adv_loss_meter.update(img_loss_adv.item()) # Image Loss for Adversarial Images
        image_ladv_loss_meter.update(img_loss_ladv.item()) # Image Loss for Latent Vector based Adversarial Images
        overall_loss_meter.update(overall_loss.item()) # Overall Loss (CE only)

        if batch_idx % 50 == 0:
            inputs_path = os.path.join(vis_dir, f'{epoch}_iter_{batch_idx}_inputs.png')
            adv_image_path = os.path.join(vis_dir, f'{epoch}_iter_{batch_idx}_adv_image.png')
            adv_latent_path = os.path.join(vis_dir, f'{epoch}_iter_{batch_idx}_adv_latent.png')
            save_image(images[:8], inputs_path, nrow=8, padding=2, normalize=True, value_range=(0.0, 1.)) # ONRUN_VERIFY
            save_image(images_adv[:8], adv_image_path, nrow=8, padding=2, normalize=True, value_range=(0.0, 1.))
            save_image(images_ladv[:8], adv_latent_path, nrow=8, padding=2, normalize=True, value_range=(0.0, 1.))

        progress_bar.set_description(
            'E: [{epoch}] '
            'Img Lo: {image_loss.val:.3f} ({image_loss.avg:.3f}) '
            'Adv Lo: {image_loss_adv.val:.3f} ({image_loss_adv.avg:.3f}) '
            'LAdv Lo: {image_loss_ladv.val:.3f} ({image_loss_ladv.avg:.3f}) '
            'Ovall Loss: {overall_loss.val:.3f} ({overall_loss.avg:.3f}) '.format(
                epoch=epoch,
                image_loss=image_loss_meter,
                image_loss_adv=image_adv_loss_meter,
                image_loss_ladv=image_ladv_loss_meter,
                overall_loss=overall_loss_meter))
        
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
            images_adv = test_attacker.perturb(images, labels)
            latents_adv = test_latent_attacker.perturb(latents, labels)
            images_ladv = gan(latents_adv).detach()
            
            # here we need to preprocess the perturbed latent Vector based Adversarial Images as well before passing to the classifier
            images_ladv = transform.classifier_preprocess_layer(images_ladv) # input -> Clamps to [-1, 1], scales to [0, 1] -> returned
            
            # normalise the images, images_adv, images_ladv
            images = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images)
            images_adv = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images_adv)
            images_ladv = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(images_ladv)
            
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
            '{mode} '
            'Epoch: [{epoch}] '
            'Clean Loss: {loss_clean.val:.3f} ({loss_clean.avg:.3f}) '
            'Clean Acc: {acc_clean.val:.3f} ({acc_clean.avg:.3f}) '
            'Adv Loss: {loss_adv.val:.3f} ({loss_adv.avg:.3f}) '
            'Adv Acc: {acc_adv.val:.3f} ({acc_adv.avg:.3f}) '
            'LAdv Loss: {loss_ladv.val:.3f} ({loss_ladv.avg:.3f}) '
            'LAdv Acc: {acc_ladv.val:.3f} ({acc_ladv.avg:.3f}) '.format(mode=mode, epoch=epoch, loss_clean=loss_clean_meter, acc_clean=acc_clean_meter, loss_adv=loss_adv_meter, acc_adv=acc_adv_meter, loss_ladv=loss_ladv_meter, acc_ladv=acc_ladv_meter))

    return loss_clean_meter.avg, acc_clean_meter.avg, loss_adv_meter.avg, acc_adv_meter.avg, loss_ladv_meter.avg, acc_ladv_meter.avg


best_test_clean_acc, best_test_clean_loss = 0.0, 0.0
best_test_adv_acc, best_test_adv_loss = 0.0, 0.0
best_test_ladv_acc, best_test_ladv_loss = 0.0, 0.0

for epoch in range(start_epoch, cfg.num_epochs):
    if cfg.distributed:
        train_sampler.set_epoch(epoch)
    
    # Train
    train_overall_loss = train(epoch)
    
    # Test on test data
    test_clean_loss, test_clean_acc, test_adv_loss, test_adv_acc, test_ladv_loss, test_ladv_acc = test(epoch)
    
    # lr used
    lr = optimizer.param_groups[0]['lr']
    
    # update lr
    lr_schedule.step()
    
    # Save best model - Clean Accuarcy Based
    if test_clean_acc > best_test_clean_acc:
        best_test_clean_acc = test_clean_acc
        best_test_clean_loss = test_clean_loss
        
        # save checkpoint
        checkpoint_path = os.path.join(output_dir, f'classifier-clean.pt')
        torch.save({
            'epoch': epoch + 1,
            'state_dict': [net.delta_models[i].state_dict() for i in range(int(cfg.svgd.num_particles))],
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
            'state_dict': [net.delta_models[i].state_dict() for i in range(int(cfg.svgd.num_particles))],
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
            'state_dict': [net.delta_models[i].state_dict() for i in range(int(cfg.svgd.num_particles))],
            'optimizer': optimizer.state_dict(),
            'best_test_ladv_acc': best_test_ladv_acc,
            'best_test_ladv_loss': best_test_ladv_loss
        }, checkpoint_path)
    
    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, f'classifier-last.pt')
    torch.save({
        'epoch': epoch + 1,
        'state_dict': [net.delta_models[i].state_dict() for i in range(int(cfg.svgd.num_particles))],
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