import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np

class MaskedDataset(Dataset):
    def __init__(self, root, transform):
        self.dataset = ImageFolder(root=root, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        _, height, width = image.size()

        # Create a random mask
        mask = self.random_mask(height, width)
        masked_image = image * mask

        return image, mask, masked_image

    def random_mask(self, height, width):
        mask = torch.ones((1, height, width))
        num_shapes = np.random.randint(1, 10)
        for _ in range(num_shapes):
            shape_type = np.random.choice(['rectangle', 'circle'])
            if shape_type == 'rectangle':
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                w = np.random.randint(10, width // 2)
                h = np.random.randint(10, height // 2)
                x1 = np.clip(x, 0, width)
                x2 = np.clip(x + w, 0, width)
                y1 = np.clip(y, 0, height)
                y2 = np.clip(y + h, 0, height)
                mask[:, y1:y2, x1:x2] = 0
            else:
                center_x = np.random.randint(0, width)
                center_y = np.random.randint(0, height)
                radius = np.random.randint(10, min(width, height) // 4)
                y, x = np.ogrid[:height, :width]
                dist_from_center = (x - center_x) ** 2 + (y - center_y) ** 2
                mask[:, dist_from_center <= radius ** 2] = 0
        return mask
    