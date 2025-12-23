import torch
import torch.nn as nn
from torchvision.models import resnet18

class MultiTaskResNet(nn.Module):
    def __init__(self):
        super(MultiTaskResNet, self).__init__()
        
        # Load pretrained ResNet18
        # Note: 'pretrained=True' is deprecated in newer versions in favor of 'weights=ResNet18_Weights.DEFAULT'
        # but for compatibility with older environments (like typical student setups), we'll stick to the request or standard usage.
        # If the environment is very new, we might need to adjust, but 'pretrained=True' usually still works with a warning.
        self.backbone = resnet18(pretrained=True)
        
        # Remove fc and avgpool
        # We want the features before avgpool
        # For 224x224 input -> 512x7x7
        # For 320x320 input -> 512x10x10
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Head 1: Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, 4)
        
        # Head 2: Segmentation
        # Input: 512 x H/32 x W/32
        # Target: 1 x H x W
        self.seg_decoder = nn.Sequential(
            # 10 -> 20 (if input 320)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 20 -> 40
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 40 -> 80
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 80 -> 160
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 160 -> 320
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            # Final activation is Sigmoid as per instructions
            nn.Sigmoid()
        )

    def forward(self, x):
        # Backbone features: [B, 512, 7, 7]
        x = self.features(x)
        
        # Head 1: Classification
        x_cls = self.avgpool(x)      # [B, 512, 1, 1]
        x_cls = torch.flatten(x_cls, 1) # [B, 512]
        out_cls = self.classifier(x_cls) # [B, 4]
        
        # Head 2: Segmentation
        out_seg = self.seg_decoder(x) # [B, 1, 224, 224]
        
        return out_cls, out_seg
