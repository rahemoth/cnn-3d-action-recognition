"""2D CNN architecture for action recognition using stacked frames."""

import torch
import torch.nn as nn


class CNN2D(nn.Module):
    """
    Simple 2D CNN for action recognition.
    
    Treats temporal dimension by stacking frames as input channels.
    Input shape: (batch, num_frames * 3, height, width)
    """
    
    def __init__(self, num_classes: int = 3, num_frames: int = 8):
        """
        Args:
            num_classes: Number of action classes
            num_frames: Number of frames (used to calculate input channels)
        """
        super(CNN2D, self).__init__()
        
        self.num_frames = num_frames
        in_channels = num_frames * 3  # RGB frames stacked
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Conv block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112 -> 56x56
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56 -> 28x28
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            
            # Conv block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, num_frames, C, H, W)
            
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Reshape from (batch, num_frames, C, H, W) to (batch, num_frames*C, H, W)
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_frames * 3, x.size(3), x.size(4))
        
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        return x
