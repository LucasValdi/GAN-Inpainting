import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Discriminator(nn.Module):
    def __init__(self, image_channels, feature_maps):
        super().__init__()

        def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, final=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            ]
            if final is False:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                
            return nn.Sequential(*layers)

        def shared_block(in_channels, out_channels):
            return nn.Sequential(
                conv_block(in_channels, out_channels), # channels = 3 -> 64 // dimension = 256x256 -> 128x128 
                conv_block(out_channels, out_channels * 2), # 64 -> 128 // 128x128 -> 64x64
                conv_block(out_channels * 2, out_channels * 4), # 128 -> 256 // 64x64 -> 32x32
            )

        def local_block(in_channels, out_channels):
            return nn.Sequential(
                conv_block(in_channels, out_channels, stride=1), # 256 -> 512 // 32x32 -> 32x32
                conv_block(out_channels, 1, stride=1, final=True), # 512 -> 1 // 31x31 -> 30x30
            )

        def global_block(in_channels, out_channels):
            return nn.Sequential(
                conv_block(in_channels, out_channels), # 256 -> 512 // 32x32 -> 16x16
                conv_block(out_channels, out_channels), # 512 -> 512 // 16x16 -> 8x8
                conv_block(out_channels, out_channels), # 512 -> 512 // 8x8 -> 4x4
                conv_block(out_channels, 1, stride=1, padding=0, final=True), # 512 -> 512 // 4x4 -> 1x1
            )

        self.shared_layers = shared_block(image_channels, feature_maps)
        self.local_layers = local_block(feature_maps * 4, feature_maps * 8)
        self.global_layers = global_block(feature_maps * 4, feature_maps * 8)

    def forward(self, x):
        shared_output = self.shared_layers(x)
        local_output = self.local_layers(shared_output)
        global_output = self.global_layers(shared_output)
        return local_output, global_output

def test():
    x = torch.randn((1, 3, 256, 256))
    model = Discriminator(3, 64)
    preds_local, preds_global = model(x)
    print(preds_local.shape)
    print(preds_local)  
    print(preds_global.shape) 
    print(preds_global)

if __name__ == "__main__":
    test()