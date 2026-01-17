import torch
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import create_dataloaders
from model import get_model


def evaluate_model(model_path, data_dir, batch_size=4, split='test'):
    """
    Comprehensive evaluation of the trained model
    
    Args:
        model_path: Path to trained model checkpoint
        data_dir: Path to celebdf_processed_data
        batch_size: Batch size for evaluation
        split: Which split to evaluate ('train', 'val', or 'test')
    
    Returns:
        dict: Evaluation metrics
    """
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = get_model(freeze_cnn=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load data
    print(f"\nLoading {split} dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir, 
        batch_size=batch_size,
        num_workers=4
    )
    
    # Select the appropriate loader
    if split == 'train':
        dataloader = train_loader
    elif split == 'val':
        dataloader = val_loader
    else:
        dataloader = test_loader
    
    # Collect predictions and labels
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    print(f"\nEvaluating on {split} set...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            
            # Get model outputs (logits)
            logits = model(inputs).squeeze()
            
            # Convert to probabilities
            probabilities = torch.sigmoid(logits)
            
            # Get predictions (threshold = 0.5)
            predictions = (probabilities > 0.5).float()
            
            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    auc_roc = roc_auc_score(all_labels, all_probabilities)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Confidence statistics
    real_confidences = all_probabilities[all_labels == 0]
    fake_confidences = all_probabilities[all_labels == 1]
    
    # Print results
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS - {split.upper()} SET")
    print("="*60)
    
    print("\n📊 Classification Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc_roc:.4f}")
    
    print("\n🎯 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Real    Fake")
    print(f"  Actual Real  {conf_matrix[0,0]:4d}    {conf_matrix[0,1]:4d}")
    print(f"       Fake  {conf_matrix[1,0]:4d}    {conf_matrix[1,1]:4d}")
    
    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print("\n📈 Additional Metrics:")
    print(f"  True Positives:  {tp}")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  Specificity:     {specificity:.4f}")
    
    print("\n💯 Confidence Statistics:")
    print(f"  Overall:")
    print(f"    Mean:   {np.mean(all_probabilities):.4f}")
    print(f"    Std:    {np.std(all_probabilities):.4f}")
    print(f"    Min:    {np.min(all_probabilities):.4f}")
    print(f"    Max:    {np.max(all_probabilities):.4f}")
    
    print(f"\n  Real Videos (should be low confidence):")
    print(f"    Mean:   {np.mean(real_confidences):.4f}")
    print(f"    Std:    {np.std(real_confidences):.4f}")
    print(f"    Median: {np.median(real_confidences):.4f}")
    
    print(f"\n  Fake Videos (should be high confidence):")
    print(f"    Mean:   {np.mean(fake_confidences):.4f}")
    print(f"    Std:    {np.std(fake_confidences):.4f}")
    print(f"    Median: {np.median(fake_confidences):.4f}")
    
    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probabilities)
    
    plt.figure(figsize=(12, 5))
    
    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {split.upper()} Set')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Confidence Distribution
    plt.subplot(1, 2, 2)
    plt.hist(real_confidences, bins=30, alpha=0.5, label='Real Videos', color='green', density=True)
    plt.hist(fake_confidences, bins=30, alpha=0.5, label='Fake Videos', color='red', density=True)
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title(f'Confidence Distribution - {split.upper()} Set')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f'evaluation_{split}_metrics.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plots saved to: {plot_path}")
    plt.show()
    
    print("\n" + "="*60)
    
    # Return metrics dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'confusion_matrix': conf_matrix,
        'confidence_stats': {
            'overall_mean': np.mean(all_probabilities),
            'overall_std': np.std(all_probabilities),
            'real_mean': np.mean(real_confidences),
            'real_std': np.std(real_confidences),
            'fake_mean': np.mean(fake_confidences),
            'fake_std': np.std(fake_confidences),
        }
    }


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = r"checkpoints\best_model.pth"
    DATA_DIR = r"c:\Users\sapta\Documents\AI_ML\DeepFake\celebdf_processed_data"
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION MODEL EVALUATION")
    print("="*60)
    
    test_metrics = evaluate_model(
        model_path=MODEL_PATH,
        data_dir=DATA_DIR,
        batch_size=4,
        split='test'
    )
    
    # Optionally evaluate on validation set too
    print("\n\n")
    val_metrics = evaluate_model(
        model_path=MODEL_PATH,
        data_dir=DATA_DIR,
        batch_size=4,
        split='val'
    )
    
    print("\n✓ Evaluation completed!")
