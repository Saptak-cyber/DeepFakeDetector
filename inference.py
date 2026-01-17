import torch
import json
import os
import numpy as np
from datetime import timedelta
from model import get_model
from preprocess import VideoPreprocessor


def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def sliding_window_smoothing(predictions, window_size=5, min_segment_length=3):
    """
    Apply sliding window smoothing to reduce noise in frame-level predictions
    
    Args:
        predictions: List of probability values (0-1) for each window
        window_size: Size of smoothing window (odd number recommended)
        min_segment_length: Minimum number of consecutive windows to form a segment
    
    Returns:
        numpy.ndarray: Smoothed predictions
    """
    if len(predictions) == 0:
        return np.array([])
    
    predictions = np.array(predictions)
    smoothed = np.zeros_like(predictions)
    
    # Apply moving average filter
    half_window = window_size // 2
    for i in range(len(predictions)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(predictions), i + half_window + 1)
        smoothed[i] = np.mean(predictions[start_idx:end_idx])
    
    return smoothed


def merge_segments(segments, time_ranges, max_gap=2.0):
    """
    Merge adjacent segments that are close together
    
    Args:
        segments: List of (index, confidence) tuples
        time_ranges: List of (start_sec, end_sec) tuples for each window
        max_gap: Maximum gap in seconds to merge segments
    
    Returns:
        list: Merged segments with format [{start_time, end_time, confidence}]
    """
    if len(segments) == 0:
        return []
    
    merged = []
    current_segment = {
        'start_idx': segments[0][0],
        'end_idx': segments[0][0],
        'confidences': [segments[0][1]]
    }
    
    for i in range(1, len(segments)):
        idx, conf = segments[i]
        
        # Check if this segment is close to the current one
        prev_end_time = time_ranges[current_segment['end_idx']][1]
        curr_start_time = time_ranges[idx][0]
        gap = curr_start_time - prev_end_time
        
        if gap <= max_gap:
            # Merge with current segment
            current_segment['end_idx'] = idx
            current_segment['confidences'].append(conf)
        else:
            # Save current segment and start new one
            merged.append(current_segment)
            current_segment = {
                'start_idx': idx,
                'end_idx': idx,
                'confidences': [conf]
            }
    
    # Add the last segment
    merged.append(current_segment)
    
    # Convert to final format
    result = []
    for seg in merged:
        start_time = time_ranges[seg['start_idx']][0]
        end_time = time_ranges[seg['end_idx']][1]
        avg_confidence = np.mean(seg['confidences'])
        
        result.append({
            "start_time": format_timestamp(start_time),
            "end_time": format_timestamp(end_time),
            "confidence": round(float(avg_confidence), 2)
        })
    
    return result


def predict_video(video_path, model_path, threshold=0.5, confidence_threshold=0.7, 
                  use_smoothing=True, smoothing_window=5, min_segment_length=3, max_merge_gap=2.0):
    """
    Predict if a video is fake and identify manipulated segments
    
    Args:
        video_path: Path to video file to analyze
        model_path: Path to trained model (.pth file)
        threshold: Threshold for fake/real classification (default: 0.5)
        confidence_threshold: Minimum confidence to report a segment (default: 0.7)
        use_smoothing: Apply sliding window smoothing (default: True)
        smoothing_window: Size of smoothing window (default: 5)
        min_segment_length: Minimum windows to form a segment (default: 3)
        max_merge_gap: Maximum gap in seconds to merge segments (default: 2.0)
    
    Returns:
        dict: Prediction results in required JSON format
    """
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = get_model(freeze_cnn=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    # Process video
    print(f"Processing video: {video_path}")
    preprocessor = VideoPreprocessor(sequence_length=30, target_fps=3, image_size=224)
    video_windows, time_ranges = preprocessor.process_video(video_path)
    
    if video_windows is None:
        return {
            "input_type": "video",
            "video_is_fake": False,
            "overall_confidence": 0.0,
            "manipulated_segments": [],
            "error": "No faces detected in video"
        }
    
    # Move to device
    video_windows = video_windows.to(device)
    
    # Predict on each window (model outputs logits)
    print(f"Analyzing {len(video_windows)} windows...")
    window_predictions = []
    
    with torch.no_grad():
        for i in range(len(video_windows)):
            window = video_windows[i].unsqueeze(0)  # Add batch dimension
            logits = model(window).squeeze().item()
            # Convert logits to probability
            probability = torch.sigmoid(torch.tensor(logits)).item()
            window_predictions.append(probability)
    
    # Apply sliding window smoothing
    if use_smoothing:
        print(f"Applying sliding window smoothing (window_size={smoothing_window})...")
        smoothed_predictions = sliding_window_smoothing(
            window_predictions, 
            window_size=smoothing_window,
            min_segment_length=min_segment_length
        )
    else:
        smoothed_predictions = np.array(window_predictions)
    
    # Determine overall fake probability
    overall_confidence = float(np.mean(smoothed_predictions))
    video_is_fake = overall_confidence > threshold
    
    # Identify manipulated segments using smoothed predictions
    candidate_segments = []
    
    # Find consecutive segments above threshold
    in_segment = False
    segment_start_idx = 0
    segment_confidences = []
    
    for i, confidence in enumerate(smoothed_predictions):
        if confidence > confidence_threshold:
            if not in_segment:
                # Start new segment
                in_segment = True
                segment_start_idx = i
                segment_confidences = [confidence]
            else:
                # Continue current segment
                segment_confidences.append(confidence)
        else:
            if in_segment:
                # End current segment if it meets minimum length
                if len(segment_confidences) >= min_segment_length:
                    candidate_segments.append((segment_start_idx, confidence))
                    for j in range(segment_start_idx, i):
                        if j < len(smoothed_predictions):
                            candidate_segments.append((j, smoothed_predictions[j]))
                in_segment = False
                segment_confidences = []
    
    # Handle last segment if still active
    if in_segment and len(segment_confidences) >= min_segment_length:
        for j in range(segment_start_idx, len(smoothed_predictions)):
            candidate_segments.append((j, smoothed_predictions[j]))
    
    # Merge adjacent segments
    manipulated_segments = merge_segments(candidate_segments, time_ranges, max_gap=max_merge_gap)
    
    # Build result
    result = {
        "input_type": "video",
        "video_is_fake": video_is_fake,
        "overall_confidence": round(overall_confidence, 2),
        "manipulated_segments": manipulated_segments
    }
    
    return result


def predict_and_save(video_path, model_path, output_json=None, threshold=0.5, confidence_threshold=0.7,
                    use_smoothing=True, smoothing_window=5, min_segment_length=3, max_merge_gap=2.0):
    """
    Predict and save results to JSON file
    
    Args:
        video_path: Path to video file
        model_path: Path to trained model
        output_json: Path to save JSON output (optional)
        threshold: Threshold for fake/real classification
        confidence_threshold: Minimum confidence to report a segment
        use_smoothing: Apply sliding window smoothing
        smoothing_window: Size of smoothing window
        min_segment_length: Minimum windows to form a segment
        max_merge_gap: Maximum gap in seconds to merge segments
    """
    # Get prediction
    result = predict_video(
        video_path, model_path, threshold, confidence_threshold,
        use_smoothing, smoothing_window, min_segment_length, max_merge_gap
    )
    
    # Print results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(json.dumps(result, indent=2))
    
    # Save to file if specified
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Results saved to: {output_json}")
    
    return result


if __name__ == "__main__":
    # Example usage
    VIDEO_PATH = r"c:\Users\sapta\Documents\AI_ML\DeepFake\celebdf_split\test\Celeb-synthesis\id0_id16_0008.mp4"
    MODEL_PATH = r"checkpoints\best_model.pth"
    OUTPUT_JSON = "prediction_result.json"
    
    # Run prediction with sliding window smoothing
    result = predict_and_save(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        output_json=OUTPUT_JSON,
        threshold=0.5,              # 0.5 = 50% probability to classify as fake
        confidence_threshold=0.7,   # Only report segments with >70% confidence
        use_smoothing=True,         # Apply sliding window smoothing
        smoothing_window=5,         # Smooth over 5 consecutive windows
        min_segment_length=3,       # Minimum 3 windows to form a segment
        max_merge_gap=2.0          # Merge segments within 2 seconds
    )
