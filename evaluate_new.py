import torch
import json
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import timedelta
import os

from dataset import create_dataloaders
from model import get_model


def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def sliding_window_smoothing(predictions, window_size=5):
    """Apply sliding window smoothing to predictions"""
    if len(predictions) == 0:
        return np.array([])
    
    predictions = np.array(predictions)
    smoothed = np.zeros_like(predictions)
    
    half_window = window_size // 2
    for i in range(len(predictions)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(predictions), i + half_window + 1)
        smoothed[i] = np.mean(predictions[start_idx:end_idx])
    
    return smoothed


def identify_manipulated_segments(predictions, confidence_threshold=0.7, 
                                  min_segment_length=3, window_duration=10):
    """
    Identify manipulated segments from window predictions
    
    Args:
        predictions: Array of confidence scores for each window
        confidence_threshold: Minimum confidence to consider as manipulated
        min_segment_length: Minimum number of consecutive windows
        window_duration: Duration of each window in seconds (30 frames @ 3fps = 10 sec)
    
    Returns:
        list: Manipulated segments with timestamps
    """
    segments = []
    in_segment = False
    segment_start = 0
    segment_confidences = []
    
    for i, confidence in enumerate(predictions):
        if confidence > confidence_threshold:
            if not in_segment:
                in_segment = True
                segment_start = i
                segment_confidences = [confidence]
            else:
                segment_confidences.append(confidence)
        else:
            if in_segment and len(segment_confidences) >= min_segment_length:
                # End segment
                start_time = segment_start * window_duration
                end_time = (segment_start + len(segment_confidences)) * window_duration
                avg_confidence = np.mean(segment_confidences)
                
                segments.append({
                    "start_time": format_timestamp(start_time),
                    "end_time": format_timestamp(end_time),
                    "confidence": round(float(avg_confidence), 2)
                })
            
            in_segment = False
            segment_confidences = []
    
    # Handle last segment
    if in_segment and len(segment_confidences) >= min_segment_length:
        start_time = segment_start * window_duration
        end_time = (segment_start + len(segment_confidences)) * window_duration
        avg_confidence = np.mean(segment_confidences)
        
        segments.append({
            "start_time": format_timestamp(start_time),
            "end_time": format_timestamp(end_time),
            "confidence": round(float(avg_confidence), 2)
        })
    
    return segments


def evaluate_with_segments(model_path, data_dir, batch_size=4, split='test', 
                           output_dir='evaluation_results'):
    """
    Comprehensive evaluation with both metrics and segment-level predictions
    
    Args:
        model_path: Path to trained model checkpoint
        data_dir: Path to celebdf_processed_data
        batch_size: Batch size for evaluation
        split: Which split to evaluate ('train', 'val', or 'test')
        output_dir: Directory to save detailed results
    
    Returns:
        dict: Evaluation metrics and results
    """
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = get_model(freeze_cnn=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load preprocessed data directly
    print(f"\nLoading {split} dataset...")
    split_dir = os.path.join(data_dir, split)
    
    categories = {
        'Celeb-real': 0,
        'YouTube-real': 0,
        'Celeb-synthesis': 1
    }
    
    # Storage for metrics
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    # Storage for detailed video results
    detailed_results = {}
    
    print(f"\nEvaluating {split} set with segment detection...")
    
    for category, true_label in categories.items():
        category_dir = os.path.join(split_dir, category)
        if not os.path.exists(category_dir):
            continue
        
        from pathlib import Path
        pt_files = list(Path(category_dir).glob('*.pt'))
        
        for pt_file in tqdm(pt_files, desc=f"  {category}"):
            try:
                # Load preprocessed data
                data = torch.load(pt_file, map_location='cpu')
                windows = data['windows']
                num_windows = data['num_windows']
                
                if num_windows == 0:
                    continue
                
                # Get predictions for each window
                window_predictions = []
                
                for i in range(0, num_windows, batch_size):
                    batch = windows[i:i+batch_size].to(device)
                    
                    with torch.no_grad():
                        logits = model(batch).squeeze()
                        if logits.dim() == 0:
                            logits = logits.unsqueeze(0)
                        probs = torch.sigmoid(logits)
                        window_predictions.extend(probs.cpu().numpy())
                
                # Apply smoothing
                smoothed_predictions = sliding_window_smoothing(window_predictions, window_size=5)
                
                # Overall video prediction
                overall_confidence = float(np.mean(smoothed_predictions))
                video_is_fake = overall_confidence > 0.5
                predicted_label = 1 if video_is_fake else 0
                
                # Identify manipulated segments
                manipulated_segments = identify_manipulated_segments(
                    smoothed_predictions,
                    confidence_threshold=0.7,
                    min_segment_length=3,
                    window_duration=10
                )
                
                # Store for metrics calculation
                all_labels.append(true_label)
                all_predictions.append(predicted_label)
                all_probabilities.append(overall_confidence)
                
                # Store detailed result
                video_name = pt_file.stem
                detailed_results[video_name] = {
                    "input_type": "video",
                    "video_name": video_name,
                    "true_label": "Fake" if true_label == 1 else "Real",
                    "video_is_fake": video_is_fake,
                    "overall_confidence": round(overall_confidence, 2),
                    "manipulated_segments": manipulated_segments,
                    "num_windows_analyzed": num_windows,
                    "is_correct": (predicted_label == true_label)
                }
                
            except Exception as e:
                print(f"\n  Error processing {pt_file.name}: {str(e)}")
                continue
    
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
    
    # Calculate additional metrics
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
    
    print(f"\n  Real Videos (should be low):")
    print(f"    Mean:   {np.mean(real_confidences):.4f}")
    print(f"    Std:    {np.std(real_confidences):.4f}")
    
    print(f"\n  Fake Videos (should be high):")
    print(f"    Mean:   {np.mean(fake_confidences):.4f}")
    print(f"    Std:    {np.std(fake_confidences):.4f}")
    
    # Save detailed video predictions
    detailed_file = os.path.join(output_dir, f'{split}_detailed_predictions.json')
    with open(detailed_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Save metrics summary
    summary = {
        'split': split,
        'total_videos': len(all_labels),
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'auc_roc': round(auc_roc, 4),
        'confusion_matrix': conf_matrix.tolist(),
        'confidence_stats': {
            'overall_mean': round(float(np.mean(all_probabilities)), 4),
            'overall_std': round(float(np.std(all_probabilities)), 4),
            'real_mean': round(float(np.mean(real_confidences)), 4),
            'real_std': round(float(np.std(real_confidences)), 4),
            'fake_mean': round(float(np.mean(fake_confidences)), 4),
            'fake_std': round(float(np.std(fake_confidences)), 4)
        }
    }
    
    summary_file = os.path.join(output_dir, f'{split}_metrics_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Plot visualizations
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
    
    plot_path = os.path.join(output_dir, f'{split}_evaluation_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plots saved to: {plot_path}")
    plt.close()
    
    # Print sample predictions
    print(f"\n📹 Sample Detailed Predictions:")
    sample_count = 0
    for video_name, result in detailed_results.items():
        if sample_count >= 3:
            break
        
        print(f"\n  Video: {video_name}")
        print(f"  True Label: {result['true_label']}")
        print(f"  Predicted: {'Fake' if result['video_is_fake'] else 'Real'}")
        print(f"  Confidence: {result['overall_confidence']}")
        print(f"  Segments: {len(result['manipulated_segments'])}")
        
        if result['manipulated_segments']:
            for seg in result['manipulated_segments'][:2]:
                print(f"    - {seg['start_time']} to {seg['end_time']} (conf: {seg['confidence']})")
        
        sample_count += 1
    
    print(f"\n✓ Results saved to:")
    print(f"  Detailed predictions: {detailed_file}")
    print(f"  Metrics summary: {summary_file}")
    print(f"  Plots: {plot_path}")
    print("="*60)
    
    return summary


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = r"checkpoints\best_model.pth"
    DATA_DIR = r"c:\Users\sapta\Documents\AI_ML\DeepFake\celebdf_processed_data"
    OUTPUT_DIR = r"evaluation_results_detailed"
    
    print("\n" + "="*60)
    print("COMPREHENSIVE DEEPFAKE EVALUATION WITH SEGMENT DETECTION")
    print("="*60)
    
    # Evaluate test set
    test_summary = evaluate_with_segments(
        model_path=MODEL_PATH,
        data_dir=DATA_DIR,
        batch_size=8,
        split='test',
        output_dir=OUTPUT_DIR
    )
    
    # Evaluate validation set
    print("\n\n")
    val_summary = evaluate_with_segments(
        model_path=MODEL_PATH,
        data_dir=DATA_DIR,
        batch_size=8,
        split='val',
        output_dir=OUTPUT_DIR
    )
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Test Accuracy: {test_summary['accuracy']*100:.2f}%")
    print(f"Val Accuracy:  {val_summary['accuracy']*100:.2f}%")
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("="*60)
