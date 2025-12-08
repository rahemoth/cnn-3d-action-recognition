"""Training loop for action recognition models."""

import os
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.cnn2d import CNN2D
from src.models.cnn3d import CNN3D
from src.data.dataset import VideoFrameDataset
from src.training.config import TrainingConfig


class Trainer:
    """Trainer class for action recognition models."""
    
    def __init__(self, config: TrainingConfig):
        """
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        if config.model_type == "cnn2d":
            self.model = CNN2D(
                num_classes=config.num_classes,
                num_frames=config.num_frames
            )
        elif config.model_type == "cnn3d":
            self.model = CNN3D(num_classes=config.num_classes)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
        
        self.model = self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.5
        )
        
        # Tracking
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, (frames, labels) in enumerate(pbar):
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(frames)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> tuple:
        """
        Validate the model.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for frames, labels in tqdm(dataloader, desc="Validation"):
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_acc': self.best_val_acc
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with accuracy: {self.best_val_acc:.2f}%")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
        print(f"\nTraining {self.config.model_type} for {self.config.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Save best model
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                
                # Save checkpoint
                if epoch % self.config.save_interval == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)
            else:
                # Save checkpoint even without validation
                if epoch % self.config.save_interval == 0:
                    self.save_checkpoint(epoch, False)
            
            # Update learning rate
            self.scheduler.step()
        
        print("\nTraining completed!")
        if val_loader is not None:
            print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
