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
import wandb 

# parse command line options
parser = argparse.ArgumentParser(description="On-manifold adv training")
parser.add_argument("--config", default="experiments/classifiers/mixed10_adv_pgd5_pgd5_sgd_sam.yml")
parser.add_argument("--resume", default="")
args = parser.parse_args()

cfg = load_config(args.config)
trainset_cfg = cfg.dataset.train
testset_cfg = cfg.dataset.test
print(cfg)

# wandb.init(project="ATTACKS_RS50_DMAT_TEST", config=cfg)
# logging.info(cfg)

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
net = net.cuda()

# set optimizers
optimizer = load_optimizer(cfg.optimizer, params=net.parameters())

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

criterion = torch.nn.CrossEntropyLoss().cuda()
net = move_to_device(net, cfg)

# set stylegan
gan_path = './stylegan_old/pretrain/stylegan_imagenet.pth'
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
    gan.eval()

    image_loss_meter = AverageMeter()
    latent_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    for batch_idx, (images, latents, labels) in enumerate(progress_bar):
        images, latents, labels = images.cuda(), latents.cuda(), labels.cuda()

        lr = cfg.optimizer.args.lr
        if lr_schedule is not None:
            lr = lr_schedule(epoch + (batch_idx + 1) / len(trainloader))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        with ctx_noparamgrad_and_eval(model):
            images_adv = image_attacker.perturb(images, labels)
            latents_adv = latent_attacker.perturb(latents, labels)

        images_ladv = gan(latents_adv).detach()

        optimizer.zero_grad()
        image_loss = criterion(net(images_adv), labels)
        latent_loss = criterion(net(images_ladv), labels)
        total_loss = 0.5 * image_loss + 0.5 * latent_loss

        total_loss.backward()
        optimizer.step()

        image_loss_meter.update(image_loss.item(), images.size(0))
        latent_loss_meter.update(latent_loss.item(), images.size(0))
        total_loss_meter.update(total_loss.item(), images.size(0))

        progress_bar.set_description(
            'Train Epoch: [{0}] | '
            'Image Loss: {i_loss.val:.3f} | '
            'Latent Loss: {l_loss.val:.3f} | '
            'Total Loss: {t_loss.val:.3f} | '
            'lr: {1:.6f}'.format(
                epoch,
                lr,
                i_loss=image_loss_meter,
                l_loss=latent_loss_meter,
                t_loss=total_loss_meter))

        if batch_idx % 50 == 0:
            inputs_path = os.path.join(vis_dir, f'{epoch}_iter_{batch_idx}_inputs.png')
            adv_image_path = os.path.join(vis_dir, f'{epoch}_iter_{batch_idx}_adv_image.png')
            adv_latent_path = os.path.join(vis_dir, f'{epoch}_iter_{batch_idx}_adv_latent.png')
            save_image(images[:8], inputs_path, nrow=8, padding=2, normalize=True, value_range=(0., 1.))
            save_image(images_adv[:8], adv_image_path, nrow=8, padding=2, normalize=True, value_range=(0., 1.))
            save_image(images_ladv[:8], adv_latent_path, nrow=8, padding=2, normalize=True, value_range=(0., 1.))

    return image_loss_meter.avg, latent_loss_meter.avg, total_loss_meter.avg

# def test(epoch):
#     progress_bar = tqdm(testloader)
#     net.eval()

#     acc_clean = AverageMeter()
#     acc_adv = AverageMeter()

#     for batch_idx, (images, _, labels) in enumerate(progress_bar):
#         images, labels = images.cuda(), labels.cuda()
#         with ctx_noparamgrad_and_eval(net):
#             images_adv = test_attacker.perturb(images, labels)

#             pred_clean = net(images).argmax(dim=1)
#             pred_adv = net(images_adv).argmax(dim=1)

#         acc_clean.update((pred_clean == labels).float().mean().item() * 100.0, images.size(0))
#         acc_adv.update((pred_adv == labels).float().mean().item() * 100.0, images.size(0))

#         progress_bar.set_description(
#             'Test Epoch: [{0}] '
#             'Clean Acc: {acc_clean.val:.3f} ({acc_clean.avg:.3f}) '
#             'Adv Acc: {acc_adv.val:.3f} ({acc_adv.avg:.3f}) '.format(epoch, acc_clean=acc_clean, acc_adv=acc_adv))

#     logging.info(f'Epoch: {epoch} | Clean: {acc_clean.avg:.2f} % | Adv: {acc_adv.avg:.2f} %')



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
        images, latents, labels = images.cuda(), latents.cuda(), labels.cuda()
        
        with ctx_noparamgrad_and_eval(model):
            images_adv = test_attacker.perturb(images, labels)
            latents_adv = test_latent_attacker.perturb(latents, labels)
            images_ladv = gan(latents_adv).detach()
            
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


for epoch in range(start_epoch, cfg.num_epochs):
    if cfg.distributed:
        train_sampler.set_epoch(epoch)
    
    # train
    train_image_loss, train_latent_loss, train_overall_loss = train(epoch)
    
    # test
    test_clean_loss, test_clean_acc, test_adv_loss, test_adv_acc, test_ladv_loss, test_ladv_acc = test(epoch)
    
    lr = optimizer.param_groups[0]['lr']

    # wandb.log({
    #     "epoch": epoch,
    #     "lr": lr,
    #     "train_adv_loss": train_image_loss,
    #     "train_ladv_loss": train_latent_loss,
    #     "train_overall_loss": train_overall_loss,
    #     "test_clean_loss": test_clean_loss,
    #     "test_clean_acc": test_clean_acc,
    #     "test_adv_loss": test_adv_loss,
    #     "test_adv_acc": test_adv_acc,
    #     "test_ladv_loss": test_ladv_loss,
    #     "test_ladv_acc": test_ladv_acc
    # })
    
    checkpoint_path = os.path.join(output_dir, f'classifier-{epoch:03d}.pt')
    torch.save({
        'epoch': epoch + 1,
        'state_dict': net.module.state_dict(),
        'optimizer': optimizer.state_dict()
    }, checkpoint_path)

    shutil.copyfile(checkpoint_path, os.path.join(output_dir, 'classifier.pt'))



