import torch
import torch.nn as nn
from torchvision import models

# 1. Robust Import
try:
    from torchvision.models import ResNet50_Weights
except ImportError:
    ResNet50_Weights = None

class MultiTaskResNet(nn.Module):
    def __init__(self, use_aux=True):
        """
        Args:
            use_aux (bool): If True, enables the auxiliary segmentation head.
        """
        super(MultiTaskResNet, self).__init__()
        self.use_aux = use_aux

        # 2. Backbone
        # Load resnet50
        if ResNet50_Weights is not None:
            weights = ResNet50_Weights.DEFAULT
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(pretrained=True)

        # Truncate: Remove the last 2 layers (avgpool, fc)
        # ResNet50 children: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        # We want everything up to layer4.
        # Converting children to list and taking all except last 2
        layers = list(self.backbone.children())[:-2]
        self.backbone = nn.Sequential(*layers)
        
        # Channels: ResNet50 output channels are 2048
        in_features = 2048

        # 3. Head 1 (Classifier)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 4)
        )

        # 4. Head 2 (Segmentation)
        if self.use_aux:
            # We need to upsample from 7x7 (at 224x224 input) to 224x224.
            # Scaling factor is 32 (224 / 7).
            # We can use 5 ConvTranspose2d layers with stride 2. (2^5 = 32).
            
            self.seg_head = nn.Sequential(
                # 7x7 -> 14x14
                nn.ConvTranspose2d(in_features, 1024, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                
                # 14x14 -> 28x28
                nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                
                # 28x28 -> 56x56
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                # 56x56 -> 112x112
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                # 112x112 -> 224x224
                nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
                # Activation: Sigmoid
                nn.Sigmoid()
            )
        else:
            self.seg_head = None

    def forward(self, x):
        # Pass through backbone
        # x shape: [B, 3, 224, 224]
        features = self.backbone(x) # Output shape: [B, 2048, 7, 7]

        # Pass through Classifier
        out_cls = self.classifier(features) # Output shape: [B, 4]

        # Condition: If self.use_aux is True, pass through Segmentation Head
        out_seg = None
        if self.use_aux and self.seg_head is not None:
            out_seg = self.seg_head(features) # Output shape: [B, 1, 224, 224]

        return out_cls, out_seg
