import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import os
from tqdm import tqdm
import time

from dataset import create_dataloaders
from model import get_model


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training if scaler is provided
        if scaler is not None:
            with autocast():
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc


def train_model(
    data_dir,
    model_type='efficientnet',
    freeze_cnn=False,
    batch_size=4,
    num_epochs=10,
    learning_rate=0.0001,
    save_dir='checkpoints',
    use_amp=True
):
    """
    Main training function
    
    Args:
        data_dir: Path to celebdf_processed_data
        model_type: 'efficientnet' or 'mobilenet'
        freeze_cnn: Whether to freeze CNN weights
        batch_size: Batch size (reduce if OOM errors)
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        save_dir: Directory to save model checkpoints
        use_amp: Use automatic mixed precision (faster on GPU)
    """
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir, 
        batch_size=batch_size,
        num_workers=2
    )
    
    # Create model
    print(f"\nInitializing {model_type} model...")
    model = get_model(model_type=model_type, freeze_cnn=freeze_cnn)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if use_amp and device == 'cuda' else None
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f"\nEpoch Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best model! Val Acc: {val_acc:.2f}%")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("="*60)
    
    return model


if __name__ == "__main__":
    # Configuration
    DATA_DIR = r"c:\Users\sapta\Documents\AI_ML\DeepFake\celebdf_processed_data"
    
    # Train the model
    model = train_model(
        data_dir=DATA_DIR,
        model_type='efficientnet',  # or 'mobilenet' for faster training
        freeze_cnn=False,            # Set True to freeze CNN weights
        batch_size=4,                # Reduce to 2 if OOM errors
        num_epochs=10,
        learning_rate=0.0001,
        save_dir='checkpoints',
        use_amp=True                 # Use mixed precision (faster on GPU)
    )
    
    print("\n✓ Model training completed!")
    print("Checkpoints saved in 'checkpoints/' directory")
