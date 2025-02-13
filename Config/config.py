import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import MaskedDataset

LEARNING_RATE = 0.0002
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0001
BATCH_SIZE = 32
IMAGE_SIZE = 256
IMAGE_CHANNELS = 3
EPOCHS = 100
FEATURE_MAPS_D = 64
FEATURE_MAPS_G = 64
LAMBDA = 10
NUM_WORKERS= 4
RESUME_TRAINING = False
CHECKPOINT_PATH = "CheckpointsRUim/checkpoint_epoch_27.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),  # Resizes the smaller side to 256 while keeping aspect ratio
        transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),  # Crops to exactly 256x256
        
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(IMAGE_CHANNELS)], [0.5 for _ in range(IMAGE_CHANNELS)]
        ),
    ]
)

dataset = MaskedDataset(root="..\\..\\Datasets\\CelebA-HQ - With Validation\\celeba_hq_train", transform=transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

dataset_val = MaskedDataset(root="..\\..\\Datasets\\CelebA-HQ - With Validation\\target", transform=transforms)
dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)