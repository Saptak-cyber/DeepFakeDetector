import torch
import torch.nn as nn
import torchvision.models as models

class ResNeXtGRUDetector(nn.Module):
    """
    ResNeXt50 + GRU model for deepfake detection
    
    Architecture:
        1. ResNeXt50 extracts spatial features from each frame
        2. GRU captures temporal patterns across the sequence
        3. Fully connected layer outputs fake probability
    
    Advantages:
        - ResNeXt50: More accurate than EfficientNet-B0
        - GRU: 25-30% faster than LSTM with similar accuracy
        - Better for 30-frame sequences (optimal for your case)
    
    Input: (Batch, 30, 3, 224, 224) - 30 frames per window
    Output: (Batch, 1) - Probability of being fake
    """
    
    def __init__(self, freeze_cnn=False):
        super(ResNeXtGRUDetector, self).__init__()
        
        # 1. Spatial Feature Extractor (ResNeXt50)
        # Using pretrained weights from ImageNet
        weights = models.ResNeXt50_32X4D_Weights.DEFAULT
        self.cnn = models.resnext50_32x4d(weights=weights)
        
        # Remove the classification head
        # ResNeXt50 outputs 2048 features
        self.cnn.fc = nn.Identity()
        
        # Option to freeze CNN for faster training
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        # 2. Temporal Analyzer (GRU - faster than LSTM)
        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        )
        
        # 3. Final Classifier (outputs logits for BCEWithLogitsLoss)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (Batch, Sequence_Length, C, H, W) -> (B, 30, 3, 224, 224)
        
        Returns:
            output: (Batch, 1) - Probability of being fake
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # FLATTEN for CNN: Stack all frames
        # (B * 30, 3, 224, 224)
        c_in = x.view(batch_size * seq_len, c, h, w)
        
        # Pass through CNN to extract spatial features
        features = self.cnn(c_in)  # Output: (B * 30, 2048)
        
        # RESHAPE for GRU: Restore sequence dimension
        # (B, 30, 2048)
        r_in = features.view(batch_size, seq_len, -1)
        
        # Pass through GRU to capture temporal patterns
        gru_out, h_n = self.gru(r_in)
        
        # Take the output of the last time step
        last_hidden_state = gru_out[:, -1, :]  # Shape: (B, 512)
        
        # Classification (output logits for BCEWithLogitsLoss)
        out = self.fc(last_hidden_state)  # (B, 1)
        
        return out


# Model selection helper
def get_model(freeze_cnn=False):
    """
    Get ResNeXt50+GRU model
    
    Args:
        freeze_cnn: Whether to freeze CNN weights (faster training)
    
    Returns:
        ResNeXtGRUDetector instance
    """
    return ResNeXtGRUDetector(freeze_cnn=freeze_cnn)
