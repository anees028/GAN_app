import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# IMPORT YOUR NEW SEPARATE MODEL FILE
from model import Generator, Discriminator, weights_init

# --- CONFIGURATION ---
# This points to the folder wrapper, NOT the images themselves directly
dataroot = "input_dataset/"  

workers = 0
batch_size = 64
image_size = 128  # Target size (your 2560x1440 images will be resized to this)
nz = 100          # Latent vector size
num_epochs = 100
lr = 0.0002
beta1 = 0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- LOAD DATA ---
# We use transforms to handle your high-res inputs
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# --- SETUP MODELS ---
netG = Generator(0).to(device)
netD = Discriminator(0).to(device)

# Initialize weights
netG.apply(weights_init)
netD.apply(weights_init)

# Optimizer Setup
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

criterion = nn.BCELoss()
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1.
fake_label = 0.

# --- TRAINING LOOP ---
print("Starting Training...")

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        
        # 1. Update Discriminator
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()

        # 2. Update Generator
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        if i % 50 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

    # Save result
    if epoch % 5 == 0:
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake, f'output_epoch_{epoch}.png', normalize=True)
        print(f"Saved output_epoch_{epoch}.png")