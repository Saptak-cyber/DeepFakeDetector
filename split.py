import os
import random
import shutil
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Base directory
base_dir = r"c:\Users\sapta\Documents\AI_ML\DeepFake\CelebDf_new"
output_dir = r"c:\Users\sapta\Documents\AI_ML\DeepFake\celebdf_split"

# Get all video files
celeb_real_dir = os.path.join(base_dir, "Celeb-real")
youtube_real_dir = os.path.join(base_dir, "YouTube-real")
celeb_synthesis_dir = os.path.join(base_dir, "Celeb-synthesis")

# Collect all real videos with full paths
celeb_real_videos = [(os.path.join(celeb_real_dir, f), "Celeb-real", f) for f in os.listdir(celeb_real_dir) if f.endswith('.mp4')]
youtube_real_videos = [(os.path.join(youtube_real_dir, f), "YouTube-real", f) for f in os.listdir(youtube_real_dir) if f.endswith('.mp4')]
all_real_videos = celeb_real_videos + youtube_real_videos

# Collect all fake videos with full paths
fake_videos = [(os.path.join(celeb_synthesis_dir, f), "Celeb-synthesis", f) for f in os.listdir(celeb_synthesis_dir) if f.endswith('.mp4')]

# Shuffle the lists
random.shuffle(all_real_videos)
random.shuffle(fake_videos)

# Split real videos: 280 train, 60 val, 60 test
real_train = all_real_videos[:280]
real_val = all_real_videos[280:340]
real_test = all_real_videos[340:400]

# Split fake videos: 280 train, 60 val, 60 test
fake_train = fake_videos[:280]
fake_val = fake_videos[280:340]
fake_test = fake_videos[340:400]
fake_remaining = fake_videos[400:]  # 395 videos not used in this split

print(f"Real videos - Train: {len(real_train)}, Val: {len(real_val)}, Test: {len(real_test)}")
print(f"Fake videos - Train: {len(fake_train)}, Val: {len(fake_val)}, Test: {len(fake_test)}")
print(f"Fake videos remaining (not used): {len(fake_remaining)}")

# Create output directory structure
for split in ['train', 'val', 'test']:
    for folder in ['Celeb-real', 'YouTube-real', 'Celeb-synthesis']:
        folder_path = os.path.join(output_dir, split, folder)
        os.makedirs(folder_path, exist_ok=True)

print(f"\nCreated directory structure at: {output_dir}")

# Function to copy videos to the appropriate folder
def copy_videos(video_list, split_name):
    count = 0
    for src_path, category, filename in video_list:
        dst_path = os.path.join(output_dir, split_name, category, filename)
        shutil.copy2(src_path, dst_path)
        count += 1
        if count % 50 == 0:
            print(f"  Copied {count}/{len(video_list)} videos to {split_name}...")
    print(f"  Completed copying {count} videos to {split_name}")

# Copy videos to train folder
print("\nCopying videos to train folder...")
copy_videos(real_train, 'train')
copy_videos(fake_train, 'train')

# Copy videos to val folder
print("\nCopying videos to val folder...")
copy_videos(real_val, 'val')
copy_videos(fake_val, 'val')

# Copy videos to test folder
print("\nCopying videos to test folder...")
copy_videos(real_test, 'test')
copy_videos(fake_test, 'test')

# Print statistics
print("\n=== Split Statistics ===")
print(f"Total Real Videos: {len(all_real_videos)}")
print(f"  - Celeb-real: {len(celeb_real_videos)}")
print(f"  - YouTube-real: {len(youtube_real_videos)}")
print(f"Total Fake Videos: {len(fake_videos)}")
print(f"\nTrain Set: {len(real_train) + len(fake_train)} videos ({len(real_train)} real, {len(fake_train)} fake)")
print(f"Val Set: {len(real_val) + len(fake_val)} videos ({len(real_val)} real, {len(fake_val)} fake)")
print(f"Test Set: {len(real_test) + len(fake_test)} videos ({len(real_test)} real, {len(fake_test)} fake)")
print(f"Unused fake videos: {len(fake_remaining)}")

print(f"\n✓ All videos copied to: {output_dir}")