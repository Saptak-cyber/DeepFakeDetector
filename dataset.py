import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os

class DeepfakeWindowDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        """
        Dataset for loading preprocessed deepfake windows
        
        Args:
            root_dir: Base directory (e.g., 'celebdf_processed_data')
            split: 'train', 'val', or 'test'
        
        Directory structure:
            root_dir/
                train/
                    Celeb-real/      -> Label 0 (Real)
                    YouTube-real/    -> Label 0 (Real)
                    Celeb-synthesis/ -> Label 1 (Fake)
        """
        self.samples = []  # (file_path, window_index, label)
        
        split_dir = os.path.join(root_dir, split)
        
        # 1. Load Real Videos (Label = 0.0)
        # Celeb-real
        real_celeb_files = glob.glob(os.path.join(split_dir, "Celeb-real", "*.pt"))
        for f in real_celeb_files:
            data = torch.load(f, map_location='cpu', weights_only=False)
            num_windows = data['num_windows']
            # Only add windows that have exactly 30 frames
            for i in range(num_windows):
                if data['windows'][i].shape[0] == 30:
                    self.samples.append((f, i, 0.0))
        
        # YouTube-real
        real_youtube_files = glob.glob(os.path.join(split_dir, "YouTube-real", "*.pt"))
        for f in real_youtube_files:
            data = torch.load(f, map_location='cpu', weights_only=False)
            num_windows = data['num_windows']
            # Only add windows that have exactly 30 frames
            for i in range(num_windows):
                if data['windows'][i].shape[0] == 30:
                    self.samples.append((f, i, 0.0))
        
        # 2. Load Fake Videos (Label = 1.0)
        # Celeb-synthesis
        fake_files = glob.glob(os.path.join(split_dir, "Celeb-synthesis", "*.pt"))
        for f in fake_files:
            data = torch.load(f, map_location='cpu', weights_only=False)
            num_windows = data['num_windows']
            # Only add windows that have exactly 30 frames
            for i in range(num_windows):
                if data['windows'][i].shape[0] == 30:
                    self.samples.append((f, i, 1.0))
        
        print(f"{split.upper()} Dataset: {len(self.samples)} total windows "
              f"({len(real_celeb_files)} Celeb-real + {len(real_youtube_files)} YouTube-real + "
              f"{len(fake_files)} Celeb-synthesis videos)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, window_idx, label = self.samples[idx]
        
        # Load the file
        data = torch.load(path, map_location='cpu', weights_only=False)
        
        # Extract the specific window: Shape (30, 3, 224, 224)
        window_tensor = data['windows'][window_idx]
        
        # Normalize to [0, 1] if needed
        if window_tensor.max() > 1.0:
            window_tensor = window_tensor / 255.0
        
        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return window_tensor, label_tensor


# Utility function to create dataloaders
def create_dataloaders(data_dir, batch_size=8, num_workers=2):
    """
    Create train, val, and test dataloaders
    
    Args:
        data_dir: Path to celebdf_processed_data directory
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = DeepfakeWindowDataset(data_dir, split='train')
    val_dataset = DeepfakeWindowDataset(data_dir, split='val')
    test_dataset = DeepfakeWindowDataset(data_dir, split='test')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
