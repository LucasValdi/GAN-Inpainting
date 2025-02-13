from generator import Generator
from discriminator import Discriminator
import config
import matplotlib.pyplot as plt
import os 
import numpy as np
import torch
import random
from torchvision.utils import save_image
from thop import profile

generator = Generator(image_channels=config.IMAGE_CHANNELS, feature_maps=config.FEATURE_MAPS_G).to(config.DEVICE)
generator.load_state_dict(torch.load("Models\\generator.pth", map_location=config.DEVICE, weights_only=True))
generator.eval()

discriminator = Discriminator(image_channels=config.IMAGE_CHANNELS, feature_maps=config.FEATURE_MAPS_D).to(config.DEVICE)
discriminator.load_state_dict(torch.load("Models\\discriminator.pth", map_location=config.DEVICE, weights_only=True))
discriminator.eval()

# gen_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
# disc_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
# print(f"Trainable parameters in Generator: {gen_params}")
# print(f"Trainable parameters in Discriminator: {disc_params}")

# def visualize_discriminator_output(local_out, global_out, idx):
#     # Convert outputs to numpy arrays
#     local_map = local_out[0, 0].detach().cpu().numpy()
#     global_map = global_out[0, 0].detach().cpu().numpy()

#     # Map to colors: Green intensity for real, Red intensity for fake
#     def map_to_color(output):
#         h, w = output.shape
#         color_map = np.zeros((h, w, 3), dtype=np.float32)  # RGB image
        
#         # Intensity proportional to confidence
#         green_intensity = output  # Higher values mean more real
#         red_intensity = 1.0 - output  # Lower values mean more real
        
#         # Assign to RGB channels
#         color_map[:, :, 1] = green_intensity  # Green channel
#         color_map[:, :, 0] = red_intensity   # Red channel
#         return color_map

#     local_color_map = map_to_color(local_map)
#     global_color_map = map_to_color(global_map)

#     # Plot the results
#     fig = plt.figure(figsize=(10, 4))

#     plt.subplot(1, 2, 1)
#     plt.title("Local Discriminator Output")
#     plt.imshow(local_color_map)
#     plt.axis('off')

#     plt.subplot(1, 2, 2)
#     plt.title("Global Discriminator Output")
#     plt.imshow(global_color_map)
#     plt.axis('off')

#     plt.tight_layout()
#     plt.show()

#     fig.savefig(os.path.join("output_test", f"discriminator_result_{idx}.png"))
#     plt.close(fig)

def denormalize(tensor):
    return (tensor + 1) / 2

def visualize_inpainting(generator, discriminator, dataloader, device, num_samples=5):
    data_iter = iter(dataloader)
    try:
        real_images, masks, masked_images  = next(data_iter)
    except StopIteration:
        print("Dataset is empty.")
        return
    
    total_samples = masked_images.size(0)
    indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    for idx in indices:
        masked_img = masked_images[idx].unsqueeze(0).to(device)
        mask = masks[idx].unsqueeze(0).to(device)
        real_img = real_images[idx].to(device)
        
        with torch.no_grad():
            generated_img = generator(masked_img)
        
        inpainted_img = masked_img * mask + generated_img * (1 - mask)

        with torch.no_grad():
            local_out, global_out = discriminator(inpainted_img)
        
        # visualize_discriminator_output(local_out, global_out, idx)

        real_img_vis = denormalize(real_img.cpu())
        mask_vis = mask.cpu().squeeze(0).squeeze(0).numpy()
        masked_img_vis = denormalize((masked_img * mask).squeeze(0).cpu())
        inpainted_img_vis = denormalize(inpainted_img.squeeze(0).cpu())
        
        real_img_np = real_img_vis.permute(1, 2, 0).numpy()
        masked_img_np = masked_img_vis.permute(1, 2, 0).numpy()
        inpainted_img_np = inpainted_img_vis.permute(1, 2, 0).numpy()
        
        mask_rgb = np.repeat(mask_vis[:, :, np.newaxis], 3, axis=2)
        
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(real_img_np)
        axs[0].set_title("Original Image")
        axs[0].axis('off')
        
        axs[1].imshow(mask_rgb, cmap='gray')
        axs[1].set_title("Mask")
        axs[1].axis('off')
        
        axs[2].imshow(masked_img_np)
        axs[2].set_title("Masked Image")
        axs[2].axis('off')
        
        axs[3].imshow(inpainted_img_np)
        axs[3].set_title("Inpainted Image")
        axs[3].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        fig.savefig(os.path.join("output_test", f"inpainting_result_{idx}.png"))
        plt.close(fig)

def test():
    visualize_inpainting(generator, discriminator, config.dataloader_val, config.DEVICE, num_samples=10)

if __name__ == "__main__":
    test()