import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class ResidualBlock(nn.Module):
    """
    Inspired by https://github.com/aladdinpersson/Machine-Learning-Collection/
    and Isensee et al Improved UNet
    """
    def __init__(self, in_channels, out_channels, dropout_p=0.2):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.drop = nn.Dropout(dropout_p)
        self.res = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.res(x)
        out = self.norm1(x)
        out = self.act1(out)
        out = self.drop(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.act2(out)
        out = self.drop(out)
        out = self.conv2(out)

        out = out + residual
        return out
        
class UNet(nn.Module):
    """
    Inspired by https://github.com/aladdinpersson/Machine-Learning-Collection/
    and Isensee et al Improved UNet
    """
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512], dropout_p=0.2):
        super().__init__()

        self.upsample = nn.ModuleList()
        self.upblock = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # contracting path
        for feature in features:
            self.downs.append(ResidualBlock(in_channels=in_channels, out_channels=feature, dropout_p=dropout_p))
            in_channels = feature 

        # expanding path
        for feature in reversed(features):
            self.upsample.append(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )

            self.upblock.append(
                ResidualBlock(feature * 3, feature, dropout_p=dropout_p) # add the double conv at each layer
            )

        # bottleneck
        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2, dropout_p=dropout_p)
        self.out = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # deep supervision
        self.deep_supervision_heads = nn.ModuleList([
            nn.Conv2d(feature, out_channels, kernel_size=1)
            for feature in reversed(features[1:])  # exclude the final output
        ])

    def forward(self, x):
        input_size = x.shape[2:]  
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # perform bottleneck operation
        x = self.bottleneck(x)
        
        # Reverse skips for decoder
        skip_connections = skip_connections[::-1]       

        # Decoder
        deep_outputs = []  
        for idx, convTranspose in enumerate(self.upsample):
            x = convTranspose(x)
            upDoubleConv = self.upblock[idx]

            # resize if skip connection and upsample size mismatch
            if x.shape != skip_connections[idx].shape:
                x = TF.resize(x, size=skip_connections[idx].shape[2:])

            # add skip connection to channel dim + upsample
            x = upDoubleConv(torch.cat((skip_connections[idx], x), dim=1))

            # add deep supervision
            if idx < len(self.deep_supervision_heads):
                deep_pred = self.deep_supervision_heads[idx](x)
                deep_pred = F.interpolate(deep_pred, size=input_size, mode='bilinear', align_corners=True)
                deep_outputs.append(deep_pred)

        output = self.out(x)

        if self.training:
            return [output] + deep_outputs
        else:
            return [output]
        

class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None, dice_weight=1, ce_weight=1):
        super().__init__()
        self.class_weights = class_weights
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.offset = 1e-6 # prevent zero division

    def forward(self, preds, targets):
        # Dice
        probs = F.softmax(preds, dim=1)

        # target resizing to output if needed
        if probs.shape != targets.shape:
            targets = F.interpolate(targets, size=probs.shape[2:], mode='nearest')

        # flatten to compute per class sums
        B, C, H, W = probs.shape
        probs_flat = probs.view(B, C, -1)
        targets_flat = targets.view(B, C, -1)

        # calculate dice loss
        intersection = (probs_flat * targets_flat).sum(-1)
        union = probs_flat.sum(-1) + targets_flat.sum(-1)
        dice_per_class = (2.0 * intersection + self.offset) / (union + self.offset)
        dice_loss_per_class = 1 - dice_per_class.mean(0) 

        # apply weighting if specified
        if self.class_weights is None:
            # average across all classes
            dice_loss = dice_loss_per_class.mean()
        else:
            # compute a weighted average
            dice_loss = (dice_loss_per_class * self.class_weights).sum() / self.class_weights.sum()

        # CE part
        ce_loss = F.cross_entropy(preds, torch.argmax(targets, dim=1), weight=self.class_weights)

        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


