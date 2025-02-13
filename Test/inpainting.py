from generator import Generator
from discriminator import Discriminator
import config
import matplotlib.pyplot as plt
import os 
import numpy as np
import torch
from torchvision.utils import save_image

generator = Generator(image_channels=config.IMAGE_CHANNELS, feature_maps=config.FEATURE_MAPS_G).to(config.DEVICE)
generator.load_state_dict(torch.load("Models\\generator.pth", map_location=config.DEVICE, weights_only=True))
generator.eval()

discriminator = Discriminator(image_channels=config.IMAGE_CHANNELS, feature_maps=config.FEATURE_MAPS_D).to(config.DEVICE)
discriminator.load_state_dict(torch.load("Models\\discriminator.pth", map_location=config.DEVICE, weights_only=True))
discriminator.eval()

def visualize_inpainting(generator, discriminator, dataloader, device):
    generator.eval()
    discriminator.eval()
    
    # os.makedirs("output_test", exist_ok=True)

    for idx, (real_images, masks, masked_images) in enumerate(dataloader):
        real_images = real_images.to(device)
        masks = masks.to(device)
        masked_images = masked_images.to(device)

        # Inpainting generation
        with torch.no_grad():
            generated_img = generator(masked_images)

        inpainted_img = masked_images + generated_img * (1 - masks)

        masked_images_white = masked_images * masks + (1 - masks)

        # Save images for visualization, each one individually
        for i in range(real_images.size(0)):  # Iterate through the batch
            save_image(real_images[i].data, f'Output_real_image/real_{idx}_{i}.png', normalize=True)
            save_image(masked_images_white[i].data, f'Output_masked_image/masked_images_{idx}_{i}.png', normalize=True)
            save_image(inpainted_img[i].data, f'Output_inpainted/inpainted_{idx}_{i}.png', normalize=True)
            save_image(masks[i].data, f'Output_mask/mask_{idx}_{i}.png', normalize=True)

def test():
    visualize_inpainting(generator, discriminator, config.dataloader_val, config.DEVICE)

if __name__ == "__main__":
    test()
