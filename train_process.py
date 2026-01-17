import torch
from preprocess import VideoPreprocessor

if __name__ == "__main__":
    print("Starting video preprocessing pipeline...")
    print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Initialize preprocessor
    preprocessor = VideoPreprocessor(sequence_length=30, target_fps=3, image_size=224)
    
    # Define paths
    input_base = r"c:\Users\sapta\Documents\AI_ML\DeepFake\celebdf_split"
    output_base = r"c:\Users\sapta\Documents\AI_ML\DeepFake\celebdf_processed_data"
    
    # Process all videos
    stats = preprocessor.process_dataset(input_base, output_base)
    
    print("\n✓ All videos processed!")