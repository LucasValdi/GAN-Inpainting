import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, image_channels, feature_maps):
        super().__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, final=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            ]
            if final:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.LeakyReLU(0.2, inplace=True)) 
            return nn.Sequential(*layers)

        def encoder_block(in_channels, out_channels, conv=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            if conv:
                layers.append(conv_block(out_channels, out_channels))
            return nn.Sequential(*layers)

        def decoder_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
            )
        
        self.initial = conv_block(image_channels, feature_maps, kernel_size=5, padding=2) # 4 -> 64 // 256x256 -> 256x256
        self.downsample1 = encoder_block(feature_maps, feature_maps * 2, conv=True) # 64 -> 128 / 128 -> 128 // 256x256 -> 128x128
        self.downsample2 = encoder_block(feature_maps * 2, feature_maps * 4, conv=True) # 128 -> 256 / 256 -> 256 // 128x128 -> 64x64
        self.downsample3 = encoder_block(feature_maps * 4, feature_maps * 8, conv=True) # 256 -> 512 / 512 -> 512 // 64x64 -> 32x32
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feature_maps * 8, feature_maps * 8, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_maps * 8, feature_maps * 8, kernel_size=3, dilation=4, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_maps * 8, feature_maps * 8, kernel_size=3, dilation=8, padding=8),
            nn.ReLU(inplace=True),
        )

        self.upsample1 = decoder_block(feature_maps * 8 * 2, feature_maps * 4) # 512 -> 256 // 32x32 -> 64x64
        self.upsample2 = decoder_block(feature_maps * 4 * 2, feature_maps * 2) # 256 -> 128 // 64x64 -> 128x128
        self.upsample3 = decoder_block(feature_maps * 2 * 2, feature_maps) # 128 -> 64 // 128x128 -> 256x256
        self.final = conv_block(feature_maps, image_channels, final=True) # 64 -> 3 // 256x256 -> 256x256

    def forward(self, masked_image):
        initial = self.initial(masked_image)
        # print("Initial:", initial.shape)
        down1 = self.downsample1(initial)
        # print("down1:", down1.shape)
        down2 = self.downsample2(down1)
        # print("down2:", down2.shape)
        down3 = self.downsample3(down2)
        # print("down3:", down3.shape)
        bottleneck = self.bottleneck(down3)
        # print("bottelneck:", bottleneck.shape)
        up1 = self.upsample1(torch.cat([bottleneck, down3], dim=1))
        # print("up1:", up1.shape)
        up2 = self.upsample2(torch.cat([up1, down2], dim=1))
        # print("up2:", up2.shape)
        up3 = self.upsample3(torch.cat([up2, down1], dim=1))
        # print("up3:", up3.shape)
        final = self.final(up3)
        # print("Final:", final.shape)
        return final

def test():
    x = torch.randn((1, 3, 256, 256))
    mask = (torch.rand((1, 1, 256, 256)) > 0.5).float()
    generator = Generator(image_channels=3, feature_maps=64)
    preds = generator(x, mask)
    print(preds.shape)

if __name__ == "__main__":
    test()
