import cv2
import torch
import numpy as np
import os
from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

class VideoPreprocessor:
    def __init__(self, sequence_length=30, target_fps=3, image_size=224):
        self.seq_len = sequence_length
        self.target_fps = target_fps
        self.image_size = image_size
        
        # Initialize Face Detector (MTCNN)
        # keep_all=False -> specific focus on the main face
        # device='cuda' -> CRITICAL for speed if you have a GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mtcnn = MTCNN(image_size=self.image_size, margin=0, 
                           keep_all=False, select_largest=True, 
                           device=device, post_process=False)

    def process_video(self, video_path):
        """
        Reads video, extracts faces at 3 FPS, and groups them into windows.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return None, None

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate strict frame interval to hit exactly 3 FPS
        # e.g., if video is 30fps, we grab every 10th frame
        frame_interval = max(1, int(original_fps / self.target_fps))

        processed_faces = []
        timestamps = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Step A: Frame Sampling
            if frame_idx % frame_interval == 0:
                # Calculate the exact timestamp (in seconds)
                current_time = frame_idx / original_fps
                
                # Step B: Face Cropping
                face_tensor = self._detect_face(frame)
                
                if face_tensor is not None:
                    processed_faces.append(face_tensor)
                    timestamps.append(current_time)
                else:
                    # Optional: Handle missing faces (e.g., keep temporal flow)
                    # For hackathon, it's safer to skip frames with no clear face
                    pass 

            frame_idx += 1

        cap.release()

        # Check if we have enough frames
        if len(processed_faces) < 1:
            return None, None

        # Stack into one big tensor: (Total_Faces, 3, 224, 224)
        all_faces_tensor = torch.stack(processed_faces)
        
        # Step C: Windowing (Sliding Window)
        return self._create_windows(all_faces_tensor, timestamps)

    def _detect_face(self, frame_bgr):
        try:
            # Convert BGR (OpenCV) to RGB (PIL/MTCNN)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # mtcnn returns a tensor scaled to [0, 1] if post_process=True
            # But we often want raw pixels normalized later. 
            # Here let's assume MTCNN standard normalization (-1 to 1 or 0 to 1)
            # This returns shape (3, 224, 224)
            face = self.mtcnn(pil_img) 
            
            # Normalize to 0-1 range for standard CNNs if MTCNN didn't
            if face is not None:
                # Standard ImageNet normalization is usually done inside the model transform
                # But for simplicity, we return the tensor as is.
                return face
                
        except Exception as e:
            # Face detection failed
            return None
        return None

    def _create_windows(self, faces_tensor, timestamps):
        num_frames = faces_tensor.size(0)
        windows = []
        window_timestamps = []

        # CONFIGURATION
        # Sequence length = 30 frames (10 seconds at 3 FPS)
        # Overlap = 2 seconds (which is 6 frames at 3 FPS)
        overlap_frames = 2 * self.target_fps  # 6 frames
        stride = self.seq_len - overlap_frames # 24 frames
        
        # Safety check: Stride must be at least 1
        stride = max(1, stride)
        
        for i in range(0, num_frames - self.seq_len + 1, stride):
            # Extract window
            window = faces_tensor[i : i + self.seq_len]
            windows.append(window)
            
            # Save timestamps
            start_t = timestamps[i]
            end_t = timestamps[i + self.seq_len - 1]
            window_timestamps.append((start_t, end_t))

        # Handle the "Leftover" frames at the end
        # If the video is 15 seconds, and your last window ended at 14s, 
        # you might miss the last second. 
        # Hackathon fix: Just grab the VERY LAST 30 frames of the video explicitly.
        if num_frames >= self.seq_len and (num_frames - self.seq_len) % stride != 0:
             windows.append(faces_tensor[-self.seq_len:])
             window_timestamps.append((timestamps[-self.seq_len], timestamps[-1]))

        if len(windows) == 0:
             # Handle short videos
             return faces_tensor.unsqueeze(0), [(timestamps[0], timestamps[-1])]

        return torch.stack(windows), window_timestamps

    def save_processed_video(self, video_windows, time_ranges, output_path):
        """
        Save processed video windows and metadata to disk.
        
        Args:
            video_windows: Tensor of shape (num_windows, seq_len, 3, H, W)
            time_ranges: List of (start_time, end_time) tuples
            output_path: Path to save the .pt file (without extension)
        """
        if video_windows is None:
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as a dictionary containing both data and metadata
        save_dict = {
            'windows': video_windows,
            'timestamps': time_ranges,
            'num_windows': len(video_windows),
            'sequence_length': self.seq_len,
            'target_fps': self.target_fps,
            'image_size': self.image_size
        }
        torch.save(save_dict, f"{output_path}.pt")
        return True

    def process_dataset(self, input_base, output_base):
        """
        Process all videos from input_base and save to output_base
        
        Args:
            input_base: Base directory containing split/category/videos structure
            output_base: Base directory to save processed data
        """
        # Define splits and categories
        splits = ['train', 'val', 'test']
        categories = ['Celeb-real', 'YouTube-real', 'Celeb-synthesis']
        
        # Statistics
        total_videos = 0
        successful = 0
        failed = 0
        skipped = 0
        
        # Process each split
        for split in splits:
            print(f"\n{'='*60}")
            print(f"Processing {split.upper()} split")
            print(f"{'='*60}")
            
            for category in categories:
                input_dir = os.path.join(input_base, split, category)
                output_dir = os.path.join(output_base, split, category)
                
                # Create output directory
                os.makedirs(output_dir, exist_ok=True)
                
                # Get all mp4 files
                video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
                
                print(f"\n{category}: {len(video_files)} videos")
                
                # Process each video
                for video_file in tqdm(video_files, desc=f"{split}/{category}"):
                    total_videos += 1
                    
                    video_path = os.path.join(input_dir, video_file)
                    
                    # Define output path (same filename but .pt extension)
                    video_name = Path(video_file).stem
                    output_path = os.path.join(output_dir, video_name)
                    
                    # Check if already processed
                    if os.path.exists(f"{output_path}.pt"):
                        skipped += 1
                        continue
                    
                    try:
                        # Process video
                        video_windows, time_ranges = self.process_video(video_path)
                        
                        if video_windows is not None:
                            # Save processed data
                            success = self.save_processed_video(
                                video_windows, time_ranges, output_path
                            )
                            if success:
                                successful += 1
                            else:
                                failed += 1
                                print(f"\nFailed to save: {video_file}")
                        else:
                            failed += 1
                            print(f"\nNo faces detected in: {video_file}")
                            
                    except Exception as e:
                        failed += 1
                        print(f"\nError processing {video_file}: {str(e)}")
        
        # Print final statistics
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total videos: {total_videos}")
        print(f"Successfully processed: {successful}")
        print(f"Failed: {failed}")
        print(f"Skipped (already processed): {skipped}")
        print(f"Output directory: {output_base}")
        
        return {
            'total': total_videos,
            'successful': successful,
            'failed': failed,
            'skipped': skipped
        }