import torch
import torch.nn as nn
import torchvision.models as models

class TimeDistributedCNN(nn.Module):
    """
    CNN + LSTM model for deepfake detection
    
    Architecture:
        1. EfficientNet-B0 extracts spatial features from each frame
        2. LSTM captures temporal patterns across the sequence
        3. Fully connected layer outputs fake probability
    
    Input: (Batch, 30, 3, 224, 224) - 30 frames per window
    Output: (Batch, 1) - Probability of being fake
    """
    
    def __init__(self, freeze_cnn=False):
        super(TimeDistributedCNN, self).__init__()
        
        # 1. Spatial Feature Extractor (EfficientNet-B0)
        # Using pretrained weights from ImageNet
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.cnn = models.efficientnet_b0(weights=weights)
        
        # Remove the classification head
        # EfficientNet-B0 outputs 1280 features
        self.cnn.classifier = nn.Identity()
        
        # Option to freeze CNN for faster training
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        # 2. Temporal Analyzer (LSTM)
        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        )
        
        # 3. Final Classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        self.sigmoid = nn.Sigmoid()

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
        features = self.cnn(c_in)  # Output: (B * 30, 1280)
        
        # RESHAPE for LSTM: Restore sequence dimension
        # (B, 30, 1280)
        r_in = features.view(batch_size, seq_len, -1)
        
        # Pass through LSTM to capture temporal patterns
        lstm_out, (h_n, c_n) = self.lstm(r_in)
        
        # Take the output of the last time step
        last_hidden_state = lstm_out[:, -1, :]  # Shape: (B, 256)
        
        # Classification
        out = self.fc(last_hidden_state)  # (B, 1)
        
        return self.sigmoid(out)


class LightweightDeepfakeDetector(nn.Module):
    """
    Lightweight version with MobileNetV3 for faster training
    """
    
    def __init__(self):
        super(LightweightDeepfakeDetector, self).__init__()
        
        # 1. Lightweight CNN
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.cnn = models.mobilenet_v3_small(weights=weights)
        self.cnn.classifier = nn.Identity()
        
        # MobileNetV3-Small outputs 576 features
        # 2. LSTM
        self.lstm = nn.LSTM(
            input_size=576,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.2
        )
        
        # 3. Classifier
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # CNN feature extraction
        c_in = x.view(batch_size * seq_len, c, h, w)
        features = self.cnn(c_in)
        
        # LSTM temporal analysis
        r_in = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(r_in)
        
        # Final classification
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        
        return self.sigmoid(out)


# Model selection helper
def get_model(model_type='efficientnet', freeze_cnn=False):
    """
    Get model by type
    
    Args:
        model_type: 'efficientnet' or 'mobilenet'
        freeze_cnn: Whether to freeze CNN weights
    
    Returns:
        model instance
    """
    if model_type == 'efficientnet':
        return TimeDistributedCNN(freeze_cnn=freeze_cnn)
    elif model_type == 'mobilenet':
        return LightweightDeepfakeDetector()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
