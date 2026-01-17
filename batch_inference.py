import torch
import json
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np


def batch_predict_preprocessed(data_dir, model_path, output_dir, split='test', 
                               threshold=0.5, batch_size=4):
    """
    Run inference on preprocessed .pt files from celebdf_processed_data
    
    Args:
        data_dir: Path to celebdf_processed_data directory
        model_path: Path to trained model
        output_dir: Directory to save JSON results
        split: 'test' or 'val'
        threshold: Threshold for fake/real classification
        batch_size: Batch size for inference
    
    Returns:
        dict: Comprehensive results with metrics
    """
    from model import get_model
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = get_model(freeze_cnn=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .pt files
    split_dir = os.path.join(data_dir, split)
    categories = {
        'Celeb-real': 0,      # Label 0 = Real
        'YouTube-real': 0,     # Label 0 = Real
        'Celeb-synthesis': 1   # Label 1 = Fake
    }
    
    all_results = []
    per_video_results = {}
    
    print(f"\n{'='*60}")
    print(f"Processing {split.upper()} set...")
    print(f"{'='*60}\n")
    
    for category, true_label in categories.items():
        category_dir = os.path.join(split_dir, category)
        if not os.path.exists(category_dir):
            continue
            
        pt_files = list(Path(category_dir).glob('*.pt'))
        print(f"\nProcessing {category}: {len(pt_files)} videos")
        
        for pt_file in tqdm(pt_files, desc=f"  {category}"):
            try:
                # Load preprocessed data
                data = torch.load(pt_file, map_location='cpu')
                windows = data['windows']  # Shape: (num_windows, 30, 3, 224, 224)
                num_windows = data['num_windows']
                
                # Skip if no valid windows
                if num_windows == 0:
                    continue
                
                # Process in batches
                window_predictions = []
                for i in range(0, num_windows, batch_size):
                    batch = windows[i:i+batch_size].to(device)
                    
                    with torch.no_grad():
                        logits = model(batch).squeeze()
                        if logits.dim() == 0:  # Single sample
                            logits = logits.unsqueeze(0)
                        probs = torch.sigmoid(logits)
                        window_predictions.extend(probs.cpu().numpy())
                
                # Calculate video-level prediction
                avg_confidence = np.mean(window_predictions)
                predicted_label = 1 if avg_confidence > threshold else 0
                is_correct = (predicted_label == true_label)
                
                # Store results
                video_name = pt_file.stem
                result = {
                    'video': video_name,
                    'category': category,
                    'true_label': 'Fake' if true_label == 1 else 'Real',
                    'predicted_label': 'Fake' if predicted_label == 1 else 'Real',
                    'confidence': float(avg_confidence),
                    'num_windows': num_windows,
                    'is_correct': is_correct
                }
                
                all_results.append(result)
                per_video_results[video_name] = result
                
            except Exception as e:
                print(f"\n  Error processing {pt_file.name}: {str(e)}")
                continue
    
    # Calculate metrics
    total = len(all_results)
    correct = sum(1 for r in all_results if r['is_correct'])
    accuracy = correct / total if total > 0 else 0
    
    # Per-category metrics
    real_results = [r for r in all_results if r['true_label'] == 'Real']
    fake_results = [r for r in all_results if r['true_label'] == 'Fake']
    
    real_correct = sum(1 for r in real_results if r['is_correct'])
    fake_correct = sum(1 for r in fake_results if r['is_correct'])
    
    real_accuracy = real_correct / len(real_results) if real_results else 0
    fake_accuracy = fake_correct / len(fake_results) if fake_results else 0
    
    # Confidence statistics
    real_confidences = [r['confidence'] for r in real_results]
    fake_confidences = [r['confidence'] for r in fake_results]
    
    # Summary
    summary = {
        'split': split,
        'total_videos': total,
        'correct_predictions': correct,
        'accuracy': round(accuracy, 4),
        'threshold': threshold,
        'real_videos': {
            'count': len(real_results),
            'correct': real_correct,
            'accuracy': round(real_accuracy, 4),
            'avg_confidence': round(np.mean(real_confidences), 4) if real_confidences else 0,
            'std_confidence': round(np.std(real_confidences), 4) if real_confidences else 0
        },
        'fake_videos': {
            'count': len(fake_results),
            'correct': fake_correct,
            'accuracy': round(fake_accuracy, 4),
            'avg_confidence': round(np.mean(fake_confidences), 4) if fake_confidences else 0,
            'std_confidence': round(np.std(fake_confidences), 4) if fake_confidences else 0
        }
    }
    
    # Save results
    results_file = os.path.join(output_dir, f'{split}_predictions.json')
    with open(results_file, 'w') as f:
        json.dump(per_video_results, f, indent=2)
    
    summary_file = os.path.join(output_dir, f'{split}_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"INFERENCE RESULTS - {split.upper()} SET")
    print(f"{'='*60}")
    print(f"\nOverall:")
    print(f"  Total videos: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    
    print(f"\nReal Videos:")
    print(f"  Count: {len(real_results)}")
    print(f"  Correct: {real_correct} / {len(real_results)}")
    print(f"  Accuracy: {real_accuracy*100:.2f}%")
    print(f"  Avg Confidence: {summary['real_videos']['avg_confidence']:.4f}")
    
    print(f"\nFake Videos:")
    print(f"  Count: {len(fake_results)}")
    print(f"  Correct: {fake_correct} / {len(fake_results)}")
    print(f"  Accuracy: {fake_accuracy*100:.2f}%")
    print(f"  Avg Confidence: {summary['fake_videos']['avg_confidence']:.4f}")
    
    print(f"\n✓ Results saved to:")
    print(f"  Predictions: {results_file}")
    print(f"  Summary: {summary_file}")
    print(f"{'='*60}\n")
    
    return summary


if __name__ == "__main__":
    # Configuration
    DATA_DIR = r"c:\Users\sapta\Documents\AI_ML\DeepFake\celebdf_processed_data"
    MODEL_PATH = r"checkpoints\best_model.pth"
    OUTPUT_DIR = r"batch_inference_results"
    
    # Run inference on test set
    test_summary = batch_predict_preprocessed(
        data_dir=DATA_DIR,
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        split='test',
        threshold=0.5,
        batch_size=4
    )
    
    # Run inference on validation set
    val_summary = batch_predict_preprocessed(
        data_dir=DATA_DIR,
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        split='val',
        threshold=0.5,
        batch_size=8
    )
    
    print("\n" + "="*60)
    print("BATCH INFERENCE COMPLETE")
    print("="*60)
    print(f"Test Accuracy: {test_summary['accuracy']*100:.2f}%")
    print(f"Val Accuracy:  {val_summary['accuracy']*100:.2f}%")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("="*60)
