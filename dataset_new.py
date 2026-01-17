import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import pathlib

# ==========================================
# 1. TRAINING DATASET (Window-Level)
# Use this for training the model.
# It treats every window as an independent sample.
# ==========================================
class DeepfakeWindowDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        """
        Dataset for training: Loads individual 30-frame windows.
        Shuffles windows from different videos together.
        """
        self.samples = []  # (file_path, window_index, label)
        
        split_dir = os.path.join(root_dir, split)
        
        # Define categories and their labels
        categories = {
            'Celeb-real': 0.0,
            'YouTube-real': 0.0,
            'Celeb-synthesis': 1.0
        }
        
        for category, label in categories.items():
            cat_dir = os.path.join(split_dir, category)
            if not os.path.exists(cat_dir):
                continue
                
            pt_files = glob.glob(os.path.join(cat_dir, "*.pt"))
            
            for f in pt_files:
                try:
                    # We load metadata only to check validity, not the full tensor yet
                    data = torch.load(f, map_location='cpu', weights_only=False)
                    num_windows = data.get('num_windows', 0)
                    
                    # Add every valid window as a separate training sample
                    for i in range(num_windows):
                        if data['windows'][i].shape[0] == 30:
                            self.samples.append((f, i, label))
                except Exception as e:
                    print(f"Skipping corrupt file {f}: {e}")
        
        print(f"[{split.upper()}] WindowDataset: Found {len(self.samples)} windows.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, window_idx, label = self.samples[idx]
        
        data = torch.load(path, map_location='cpu', weights_only=False)
        window_tensor = data['windows'][window_idx]
        
        # Normalize to [0, 1]
        if window_tensor.max() > 1.0:
            window_tensor = window_tensor / 255.0
            
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return window_tensor, label_tensor


# ==========================================
# 2. EVALUATION DATASET (Video-Level)
# Use this for evaluate_new.py.
# It returns the WHOLE video so you can do timestamping.
# ==========================================
class DeepfakeVideoDataset(Dataset):
    def __init__(self, root_dir, split='test'):
        """
        Dataset for evaluation: Loads entire video files.
        Keeps windows ordered so you can perform timestamp localization.
        """
        self.video_files = [] # (file_path, label)
        split_dir = os.path.join(root_dir, split)
        
        categories = {
            'Celeb-real': 0.0,
            'YouTube-real': 0.0,
            'Celeb-synthesis': 1.0
        }
        
        for category, label in categories.items():
            cat_dir = os.path.join(split_dir, category)
            if not os.path.exists(cat_dir):
                continue
                
            pt_files = glob.glob(os.path.join(cat_dir, "*.pt"))
            for f in pt_files:
                self.video_files.append((f, label))
                
        print(f"[{split.upper()}] VideoDataset: Found {len(self.video_files)} full videos.")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        path, label = self.video_files[idx]
        
        # Load the whole file (all windows for this video)
        data = torch.load(path, map_location='cpu', weights_only=False)
        
        return {
            'windows': data['windows'],       # Shape: (N, 30, 3, 224, 224)
            'num_windows': data['num_windows'],
            'label': label,
            'video_name': os.path.basename(path)
        }


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def create_dataloaders(data_dir, batch_size=8, num_workers=2):
    """
    Standard dataloaders for TRAINING (Window-based).
    """
    train_dataset = DeepfakeWindowDataset(data_dir, split='train')
    val_dataset = DeepfakeWindowDataset(data_dir, split='val')
    # For test, we might typically use the video dataset, but for 
    # simple accuracy metrics, window-based is fine too.
    test_dataset = DeepfakeWindowDataset(data_dir, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader