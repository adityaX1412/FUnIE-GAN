import os
import math
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from imresize import imresize, resize_tensor_to_multiple_of_32
import functions  
from nets.funiegan import GeneratorFunieGAN, DiscriminatorFunieGAN
from nets.commons import VGG19_PercepLoss, Weights_Normal
from torchvision.utils import save_image


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_underwater.yaml", help="Path to config file")
    parser.add_argument("--input_dir", type=str, default="/kaggle/input/euvp-underwater/EUVP Dataset/Paired/underwater_dark", help="Input directory")
    parser.add_argument("--input_name", type=str, default="/trainA/264286_00007889.jpg", help="Input image name")
    parser.add_argument("--gt_name", type=str, default="/trainB/264286_00007889.jpg", help="Input image name")
    parser.add_argument("--nfc_init", type=int, default=64, help="Initial number of filters in conv layers")
    parser.add_argument("--min_nfc_init", type=int, default=32, help="Minimum number of filters")
    parser.add_argument("--ker_size", type=int, default=3, help="Kernel size")
    parser.add_argument("--num_layer", type=int, default=5, help="Number of layers")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    parser.add_argument("--noise_amp_init", type=float, default=0.1, help="Initial noise amplitude")
    parser.add_argument("--scale_factor_init", type=float, default=0.75, help="Scale factor for pyramid")
    parser.add_argument("--scale1", type=float, default=1.0, help="Initial scale")
    parser.add_argument("--stop_scale", type=int, default=5, help="Stop scale")
    parser.add_argument("--lr_g", type=float, default=0.0005, help="Generator learning rate")
    parser.add_argument("--lr_d", type=float, default=0.0005, help="Discriminator learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam optimizer")
    parser.add_argument("--niter", type=int, default=2000, help="Number of iterations")
    parser.add_argument("--nc_z", type=int, default=3, help="Number of channels in noise")
    parser.add_argument("--nc_im", type=int, default=3, help="Number of channels in image")
    parser.add_argument("--lambda_grad", type=float, default=0.1, help="Gradient penalty lambda")
    parser.add_argument("--not_cuda", action='store_true', help="Disable CUDA")
    parser.add_argument("--out", type=str, default="TrainedModels", help="Output directory")
    parser.add_argument("--manualSeed", type=int, default=None, help="Manual seed")
    parser.add_argument("--mode", type=str, default="train", help="Mode: train or random_samples")
    parser.add_argument('--alpha',type=float, help='reconstruction loss weight',default=10)
    
    args = parser.parse_args()
    
    # Try to load config file if it exists
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Update args with config values
        for key, value in config.items():
            setattr(args, key, value)
    
    # Set device
    if args.not_cuda:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Set manual seed if provided
    if args.manualSeed is not None:
        torch.manual_seed(args.manualSeed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.manualSeed)
    
    return args


def train_single_image_with_funiegan(opt):
    """Train FunieGAN on a single image using multi-scale approach"""
    print(f"Training on device: {opt.device}")
    print(f"Input image: {os.path.join(opt.input_dir, opt.input_name)}")
    print(f"GT image: {os.path.join(opt.input_dir, opt.gt_name)}")
    
    # Read and preprocess input image (distorted)
    real_ = functions.read_image(opt)
    real = imresize(real_, opt.scale1, opt)
    real_ = resize_tensor_to_multiple_of_32(real_, opt)
    reals = []
    reals = functions.creat_reals_pyramid(real, reals, opt)
    
    # Read and preprocess GT image (enhanced)
    opt.input_name = opt.gt_name  # Temporarily change to read GT
    gt_ = functions.read_image(opt)
    gt = imresize(gt_, opt.scale1, opt)
    gt_ = resize_tensor_to_multiple_of_32(gt_, opt)
    gts = []
    gts = functions.creat_reals_pyramid(gt, gts, opt)
    opt.input_name = opt.input_name.replace(opt.gt_name, opt.input_name)  # Restore original input name
    
    print(f"Created pyramid with {len(reals)} scales")
    for i, r in enumerate(reals):
        print(f"Scale {i}: {r.shape}")

    # Initialize lists to store generators, noise, and noise amplitudes
    Gs, Zs, NoiseAmp = [], [], []
    #in_s = torch.full_like(reals[0], 0, device=opt.device)
    in_s = real_
    # Train at each scale
    for scale_num in range(opt.stop_scale + 1):
        print(f"\n=== Training Scale {scale_num} ===")
        
        # Adjust number of filters based on scale
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        
        print(f"nfc: {opt.nfc}, min_nfc: {opt.min_nfc}")

        # Create output directory for this scale
        opt.outf = os.path.join(functions.generate_dir2save(opt), str(scale_num))
        os.makedirs(opt.outf, exist_ok=True)
        print(f"Output directory: {opt.outf}")

        # Get real image (distorted) and GT at current scale
        real = reals[scale_num]  # distorted image
        gt = gts[scale_num]      # ground truth enhanced image
        opt.nzx, opt.nzy = real.shape[2], real.shape[3]
        print(f"Real image shape: {real.shape}")
        print(f"GT image shape: {gt.shape}")

        # Initialize networks
        generator = GeneratorFunieGAN(opt.nc_im, opt.nc_im).to(opt.device)
        discriminator = DiscriminatorFunieGAN(opt.nc_im).to(opt.device)
        
        # Apply weight initialization
        generator.apply(Weights_Normal)
        discriminator.apply(Weights_Normal)

        # Initialize optimizers
        optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

        # Initialize loss functions
        mse = nn.MSELoss().to(opt.device)
        l1 = nn.L1Loss().to(opt.device)
        perceptual = VGG19_PercepLoss().to(opt.device)

        # Initialize padding
        pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        m_noise = nn.ZeroPad2d(pad_noise)
        m_image = nn.ZeroPad2d(pad_noise)

        # Initialize noise
        fixed_noise = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
        z_opt = torch.full_like(fixed_noise, 0)
        z_opt = m_noise(z_opt)

        # Training loop
        for epoch in range(opt.niter):
            # Generate noise
            noise_ = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
            noise_ = m_noise(noise_)

            # Handle first scale differently
            if scale_num == 0:
                prev = m_image(real)
                #z_prev = m_noise(noise_)
                #opt.noise_amp = 0.05  # minimal noise for robustness
                #noise = opt.noise_amp * z_prev + prev
                noise = prev
            else:
                # Generate previous scale output
                prev = functions.draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
                prev = m_image(prev)
                
                # Calculate noise amplitude based on reconstruction error
                z_prev = functions.draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rec', m_noise, m_image, opt)
                real, z_prev = functions.align_tensors(real, z_prev)
                rmse = torch.sqrt(mse(real, z_prev))
                opt.noise_amp = opt.noise_amp_init * rmse
                z_prev = m_image(z_prev)

            # Create input noise - ensure tensors have matching dimensions
            if prev.shape != noise_.shape:
                # Resize prev to match noise_ dimensions
                prev = torch.nn.functional.interpolate(prev, size=(noise_.shape[2], noise_.shape[3]), mode='bilinear', align_corners=False)
            
            noise = prev

            # =================
            # Train Discriminator
            # =================
            discriminator.zero_grad()
            
            # Real loss - use GT (enhanced) image as real, distorted as condition
            real_pred = discriminator(gt, real)
            real_loss = mse(real_pred, torch.ones_like(real_pred))
            
            # Fake loss - use generated image as fake, distorted as condition
            fake = generator(noise.detach())
            fake_pred = discriminator(fake.detach(), real)
            fake_loss = mse(fake_pred, torch.zeros_like(fake_pred))
            
            # Total discriminator loss
            loss_D = 0.5 * (real_loss + fake_loss)
            
            # Add gradient penalty if specified
            if hasattr(opt, 'lambda_grad') and opt.lambda_grad > 0:
                gradient_penalty = functions.calc_gradient_penalty(discriminator, gt, fake, opt.lambda_grad, opt.device)
                loss_D += opt.lambda_grad * gradient_penalty
            
            loss_D.backward()
            optimizer_D.step()

            # =================
            # Train Generator
            # =================
            generator.zero_grad()
            
            # Generate fake image with same padding handling
            h, w = noise.shape[2], noise.shape[3]
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            
            if pad_h > 0 or pad_w > 0:
                noise_padded = torch.nn.functional.pad(noise, (0, pad_w, 0, pad_h), mode='reflect')
                fake = generator(noise_padded)
                fake = fake[:, :, :h, :w]
            else:
                fake = generator(noise)
                
            fake_pred = discriminator(fake, real)

            if fake.shape[2:] != gt.shape[2:]:
                h = min(fake.shape[2], gt.shape[2])
                w = min(fake.shape[3], gt.shape[3])
                fake = fake[:, :, :h, :w]
                gt_resized = gt[:, :, :h, :w]
            else:
                gt_resized = gt

            # Adversarial loss
            loss_adv = mse(fake_pred, torch.ones_like(fake_pred))
            
            # L1 loss - compare generated with GT (enhanced)
            loss_l1 = l1(fake, gt_resized)
            
            # Perceptual loss - compare generated with GT (enhanced)
            loss_vgg = perceptual(fake, gt_resized)
            
            # Total generator loss
            loss_G = loss_adv + 10 * loss_l1 + 3 * loss_vgg
            
            loss_G.backward()
            optimizer_G.step()

            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{opt.niter}: "
                      f"D_loss: {loss_D.item():.4f}, "
                      f"G_loss: {loss_G.item():.4f}, "
                      f"Adv: {loss_adv.item():.4f}, "
                      f"L1: {loss_l1.item():.4f}, "
                      f"VGG: {loss_vgg.item():.4f}")

            # Save sample images
            if epoch % 500 == 0 or epoch == opt.niter - 1:
                with torch.no_grad():
                    fake_sample = generator(noise)
                    save_image(fake_sample, f"{opt.outf}/fake_epoch_{epoch}.png")
                    
                    # Save real and GT images for comparison
                    if epoch == 0:
                        save_image(real, f"{opt.outf}/real_distorted.png")
                        save_image(gt, f"{opt.outf}/gt_enhanced.png")

        # Store trained models
        Gs.append(generator.eval())
        Zs.append(z_opt)
        NoiseAmp.append(opt.noise_amp)

        # Save model checkpoints
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'z_opt': z_opt,
            'noise_amp': opt.noise_amp,
            'scale_num': scale_num,
        }, f"{opt.outf}/checkpoint.pth")
        
        print(f"Scale {scale_num} completed. Models saved to {opt.outf}")

    # Save final model
    final_model_path = os.path.join(functions.generate_dir2save(opt), "final_model.pth")
    torch.save({
        'Gs': [G.state_dict() for G in Gs],
        'Zs': Zs,
        'NoiseAmp': NoiseAmp,
        'reals': reals,
        'opt': opt,
    }, final_model_path)
    
    print(f"\nTraining completed! Final model saved to {final_model_path}")
    return Gs, Zs, reals, NoiseAmp


def generate_samples(opt, Gs, Zs, reals, NoiseAmp, num_samples=5):
    """Generate random samples using trained model"""
    print(f"\nGenerating {num_samples} random samples...")
    
    # Create output directory for samples
    samples_dir = os.path.join(functions.generate_dir2save(opt), "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    m_noise = nn.ZeroPad2d(pad_noise)
    m_image = nn.ZeroPad2d(pad_noise)
    
    in_s = torch.full_like(reals[0], 0, device=opt.device)
    
    for i in range(num_samples):
        print(f"Generating sample {i+1}/{num_samples}")
        
        # Generate random sample
        sample = functions.draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
        
        # Save sample
        save_image(sample, f"{samples_dir}/random_sample_{i+1}.png")
    
    print(f"Samples saved to {samples_dir}")


def main():
    """Main training function"""
    opt = get_config()
    
    print("=" * 50)
    print("FunieGAN Training Script")
    print("=" * 50)
    print(f"Configuration:")
    for key, value in vars(opt).items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    if opt.mode == 'train':
        # Train the model
        Gs, Zs, reals, NoiseAmp = train_single_image_with_funiegan(opt)
        
        # Generate some samples
        generate_samples(opt, Gs, Zs, reals, NoiseAmp, num_samples=5)
        
    elif opt.mode == 'random_samples':
        # Load trained model and generate samples
        final_model_path = os.path.join(functions.generate_dir2save(opt), "final_model.pth")
        if os.path.exists(final_model_path):
            checkpoint = torch.load(final_model_path, map_location=opt.device)
            
            # Reconstruct generators
            Gs = []
            for i, state_dict in enumerate(checkpoint['Gs']):
                G = GeneratorFunieGAN(opt.nc_im, opt.nc_im).to(opt.device)
                G.load_state_dict(state_dict)
                G.eval()
                Gs.append(G)
            
            Zs = checkpoint['Zs']
            NoiseAmp = checkpoint['NoiseAmp']
            reals = checkpoint['reals']
            
            generate_samples(opt, Gs, Zs, reals, NoiseAmp, num_samples=10)
        else:
            print(f"No trained model found at {final_model_path}")
            print("Please train the model first with --mode train")
    
    else:
        print(f"Unknown mode: {opt.mode}")
        print("Available modes: train, random_samples")


if __name__ == '__main__':
    main()
