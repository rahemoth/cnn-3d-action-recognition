"""3D CNN architecture for spatiotemporal action recognition."""

import torch
import torch.nn as nn


class CNN3D(nn.Module):
    """
    Lightweight 3D CNN for action recognition.
    
    Processes video as 5D tensor with temporal dimension.
    Input shape: (batch, channels, num_frames, height, width)
    """
    
    def __init__(self, num_classes: int = 3):
        """
        Args:
            num_classes: Number of action classes
        """
        super(CNN3D, self).__init__()
        
        # 3D Convolutional layers
        self.conv_layers = nn.Sequential(
            # Conv3D block 1
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),  # Keep temporal, downsample spatial
            
            # Conv3D block 2
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),  # Downsample temporal and spatial
            
            # Conv3D block 3
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),  # Downsample temporal and spatial
            
            # Conv3D block 4
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),  # Downsample temporal and spatial
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
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
        # Reshape from (batch, num_frames, C, H, W) to (batch, C, num_frames, H, W)
        batch_size = x.size(0)
        x = x.permute(0, 2, 1, 3, 4)
        
        # 3D Convolutional layers
        x = self.conv_layers(x)
        
        # Global average pooling
        x = self.global_pool(x)
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        return x
