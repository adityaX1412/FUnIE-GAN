"""
 > Training pipeline for FUnIE-GAN (paired) model
   * Paper: arxiv.org/pdf/1903.09766.pdf
 > Maintainer: https://github.com/xahidbuffon
"""
# py libs
import os
import sys
import yaml
import argparse
import numpy as np
from PIL import Image
# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
# local libs
from nets.commons import Weights_Normal, VGG19_PercepLoss
from nets.funiegan import GeneratorFunieGAN, DiscriminatorFunieGAN
from utils.data_utils import GetTrainingPairs, GetValImage

## get configs and training options
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="configs/train_euvp.yaml")
#parser.add_argument("--cfg_file", type=str, default="configs/train_ufo.yaml")
parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
parser.add_argument("--num_epochs", type=int, default=201, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of 1st order momentum")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of 2nd order momentum")
args = parser.parse_args()

## training params
epoch = args.epoch
num_epochs = args.num_epochs
batch_size =  args.batch_size
lr_rate, lr_b1, lr_b2 = args.lr, args.b1, args.b2 
# load the data config file
with open(args.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
# get info from config file
dataset_name = cfg["dataset_name"] 
dataset_path = cfg["dataset_path"]
channels = cfg["chans"]
img_width = cfg["im_width"]
img_height = cfg["im_height"] 
val_interval = cfg["val_interval"]
ckpt_interval = cfg["ckpt_interval"]


## create dir for model and validation data
samples_dir = os.path.join("samples/FunieGAN/", dataset_name)
checkpoint_dir = os.path.join("checkpoints/FunieGAN/", dataset_name)
results_dir = os.path.join("results/FunieGAN/", dataset_name)  # Add results directory

# Create all necessary directories
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)  # Ensure results directory exists


""" FunieGAN specifics: loss functions and patch-size
-----------------------------------------------------"""
Adv_cGAN = torch.nn.MSELoss()
L1_G  = torch.nn.L1Loss() # similarity loss (l1)
L_vgg = VGG19_PercepLoss() # content loss (vgg)
lambda_1, lambda_con = 7, 3 # 7:3 (as in paper)
patch = (1, img_height//16, img_width//16) # 16x16 for 256x256

# Initialize generator and discriminator
generator = GeneratorFunieGAN()
discriminator = DiscriminatorFunieGAN()

# see if cuda is available
if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    Adv_cGAN.cuda()
    L1_G = L1_G.cuda()
    L_vgg = L_vgg.cuda()
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

# Initialize weights or load pretrained models
if args.epoch == 0:
    generator.apply(Weights_Normal)
    discriminator.apply(Weights_Normal)
else:
    # Fixed torch.load with weights_only=True for security
    try:
        generator.load_state_dict(torch.load(
            "checkpoints/FunieGAN/%s/generator_%d.pth" % (dataset_name, args.epoch),
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            weights_only=True
        ))
        discriminator.load_state_dict(torch.load(
            "checkpoints/FunieGAN/%s/discriminator_%d.pth" % (dataset_name, epoch),
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            weights_only=True
        ))
        print ("Loaded model from epoch %d" %(epoch))
    except FileNotFoundError:
        print(f"Model files not found for epoch {args.epoch}. Starting with initialized weights.")
        generator.apply(Weights_Normal)
        discriminator.apply(Weights_Normal)
    except Exception as e:
        print(f"Error loading models: {e}. Starting with initialized weights.")
        generator.apply(Weights_Normal)
        discriminator.apply(Weights_Normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))


## Data pipeline
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    GetTrainingPairs(dataset_path, dataset_name, transforms_=transforms_),
    batch_size = batch_size,
    shuffle = True,
    num_workers = 8,
)

val_dataloader = DataLoader(
    GetValImage(dataset_path, dataset_name, transforms_=transforms_, sub_dir='validation'),
    batch_size=4,
    shuffle=True,
    num_workers=1,
)


## Training pipeline
print(f"Starting training from epoch {epoch} to {num_epochs}")
print(f"Dataset: {dataset_name}, Batch size: {batch_size}, Learning rate: {lr_rate}")

for epoch in range(epoch, num_epochs):
    epoch_d_loss = 0.0
    epoch_g_loss = 0.0
    num_batches = len(dataloader)
    
    for i, batch in enumerate(dataloader):
        # Model inputs
        imgs_distorted = Variable(batch["A"].type(Tensor))
        imgs_good_gt = Variable(batch["B"].type(Tensor))
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_distorted.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_distorted.size(0), *patch))), requires_grad=False)

        ## Train Discriminator
        optimizer_D.zero_grad()
        imgs_fake = generator(imgs_distorted)
        pred_real = discriminator(imgs_good_gt, imgs_distorted)
        loss_real = Adv_cGAN(pred_real, valid)
        pred_fake = discriminator(imgs_fake.detach(), imgs_distorted)  # Use .detach() to avoid computing gradients
        loss_fake = Adv_cGAN(pred_fake, fake)
        # Total loss: real + fake (standard PatchGAN)
        loss_D = 0.5 * (loss_real + loss_fake) * 10.0 # 10x scaled for stability
        loss_D.backward()
        optimizer_D.step()

        ## Train Generator
        optimizer_G.zero_grad()
        imgs_fake = generator(imgs_distorted)
        pred_fake = discriminator(imgs_fake, imgs_distorted)
        loss_GAN =  Adv_cGAN(pred_fake, valid) # GAN loss
        loss_1 = L1_G(imgs_fake, imgs_good_gt) # similarity loss
        loss_con = L_vgg(imgs_fake, imgs_good_gt)# content loss
        # Total loss (Section 3.2.1 in the paper)
        loss_G = loss_GAN + lambda_1 * loss_1  + lambda_con * loss_con 
        loss_G.backward()
        optimizer_G.step()

        # Accumulate losses for epoch average
        epoch_d_loss += loss_D.item()
        epoch_g_loss += loss_G.item()

        ## Print log
        if not i % 50:
            sys.stdout.write("\r[Epoch %d/%d: batch %d/%d] [DLoss: %.4f, GLoss: %.4f, AdvLoss: %.4f, L1Loss: %.4f, VGGLoss: %.4f]"
                              %(
                                epoch, num_epochs, i, len(dataloader),
                                loss_D.item(), loss_G.item(), loss_GAN.item(),
                                loss_1.item(), loss_con.item()
                               )
            )
            
        ## If at sample interval save image
        batches_done = epoch * len(dataloader) + i
        if batches_done % val_interval == 0:
            try:
                imgs = next(iter(val_dataloader))
                imgs_val = Variable(imgs["val"].type(Tensor))
                imgs_gen = generator(imgs_val)
                img_sample = torch.cat((imgs_val.data, imgs_gen.data), -2)
                sample_path = os.path.join(samples_dir, f"{batches_done}.png")
                save_image(img_sample, sample_path, nrow=5, normalize=True)
                print(f"\nSaved sample: {sample_path}")
            except Exception as e:
                print(f"\nWarning: Could not save sample at batch {batches_done}: {e}")

    # Print epoch summary
    avg_d_loss = epoch_d_loss / num_batches
    avg_g_loss = epoch_g_loss / num_batches
    print(f"\nEpoch {epoch} Summary - Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}")

    ## Save model checkpoints
    if (epoch % ckpt_interval == 0):
        try:
            gen_path = os.path.join(checkpoint_dir, f"generator_{epoch}.pth")
            disc_path = os.path.join(checkpoint_dir, f"discriminator_{epoch}.pth")
            torch.save(generator.state_dict(), gen_path)
            torch.save(discriminator.state_dict(), disc_path)
            print(f"Saved checkpoints for epoch {epoch}")
        except Exception as e:
            print(f"Warning: Could not save checkpoints for epoch {epoch}: {e}")

print("Training completed!")


