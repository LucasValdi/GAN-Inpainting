import torch.optim as optim
import torch.nn as nn
import torch
from generator import Generator
from discriminator import Discriminator
import config
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os

def save_checkpoint(epoch, generator, discriminator, optim_generator, optim_discriminator, checkpoint_dir="Checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optim_generator_state_dict": optim_generator.state_dict(),
        "optim_discriminator_state_dict": optim_discriminator.state_dict(),
    }
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(checkpoint_path, generator, discriminator, optim_generator, optim_discriminator):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    optim_generator.load_state_dict(checkpoint["optim_generator_state_dict"])
    optim_discriminator.load_state_dict(checkpoint["optim_discriminator_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Checkpoint loaded: {checkpoint_path} (Resuming from epoch {start_epoch})")
    return start_epoch

# Testar
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def main():
    generator = Generator(config.IMAGE_CHANNELS, config.FEATURE_MAPS_G).to(config.DEVICE)
    discriminator = Discriminator(config.IMAGE_CHANNELS, config.FEATURE_MAPS_D).to(config.DEVICE)  

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    adversarial_loss = nn.BCEWithLogitsLoss()
    pixelwise_loss = nn.L1Loss()

    optim_generator = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE_G, betas=(0.5, 0.999))
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE_D, betas=(0.5, 0.999))

    # Load checkpoint if specified
    start_epoch = 1
    if config.RESUME_TRAINING and os.path.exists(config.CHECKPOINT_PATH):
        start_epoch = load_checkpoint(config.CHECKPOINT_PATH, generator, discriminator, optim_generator, optim_discriminator)

    generator.train()
    discriminator.train()

    gen_to_disc_ratio = 2  # Number of generator updates for each discriminator update
    for epoch in range(start_epoch, config.EPOCHS + 1):
        loop = tqdm(config.dataloader, leave=True)

        for batch_index, (real_images, masks, masked_images) in enumerate(loop):
            real_images, masks, masked_images = real_images.to(config.DEVICE), masks.to(config.DEVICE), masked_images.to(config.DEVICE)

            # Train Generator
            optim_generator.zero_grad()
            generated_img = generator(masked_images)
            inpainted_img = masked_images + generated_img * (1 - masks)
            local_out, global_out = discriminator(inpainted_img)
            g_loss_adv = adversarial_loss(local_out, torch.ones_like(local_out)) + \
                        adversarial_loss(global_out, torch.ones_like(global_out))
            g_loss_pixel = pixelwise_loss(generated_img * (1 - masks), real_images * (1 - masks))
            g_loss = g_loss_adv + config.LAMBDA * g_loss_pixel
            g_loss.backward()
            optim_generator.step()

            # Train Discriminator
            if batch_index % 3 == 0:
                optim_discriminator.zero_grad()
                real_local_out, real_global_out = discriminator(real_images)
                fake_local_out, fake_global_out = discriminator(inpainted_img.detach())

                d_loss_real = adversarial_loss(real_local_out, torch.ones_like(real_local_out)) + \
                            adversarial_loss(real_global_out, torch.ones_like(real_global_out))
                d_loss_fake = adversarial_loss(fake_local_out, torch.zeros_like(fake_local_out)) + \
                            adversarial_loss(fake_global_out, torch.zeros_like(fake_global_out))
                d_loss = (d_loss_real + d_loss_fake) / 2
                d_loss.backward()
                optim_discriminator.step()

            # Log progress
            if batch_index % 100 == 0:
                print(f"[Epoch {epoch}/{config.EPOCHS}] [Batch {batch_index}/{len(config.dataloader)}] \
                    [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
                # Save images for visualization
                images_list = [real_images, masked_images, inpainted_img, generated_img]
                images_names = ['real', 'masked_images', 'inpainted', 'generated']
                for index, images in enumerate(images_list):
                    save_image(images.data, f'Outputs/{images_names[index]}_epoch{epoch}_batch{batch_index}.png', normalize=True)

        # Saving checkpoints
        save_checkpoint(epoch, generator, discriminator, optim_generator, optim_discriminator)

    # Saving Generator
    torch.save(generator.state_dict(), 'generator.pth')

    # Saving Discriminator
    torch.save(discriminator.state_dict(), 'discriminator.pth')

    print("Training completed!")

if __name__ == "__main__":
    main()