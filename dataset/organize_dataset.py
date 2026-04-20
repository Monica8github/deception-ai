"""
Clean dataset organization - maps 121 labeled videos to standardized format
"""

import os
import json
import glob
import shutil
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

print("=" * 70)
print("CLEANING AND ORGANIZING DATASET WITH PROPER LABELS")
print("=" * 70)

# Step 1: Read the CSV to get correct label mappings
csv_path = r"C:\Users\monic\deception_detector\dataset\raw\SATHYA_MAM\Real-life_Deception_Detection_2016\Annotation\All_Gestures_Deceptive and Truthful.csv"

print(f"\n1. Reading annotation CSV: {Path(csv_path).name}")

if HAS_PANDAS:
    df = pd.read_csv(csv_path)
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Columns: {df.columns.tolist()[:5]}... (truncated)")
    
    # Map from CSV: 'id' (filename) and 'class' (label)
    labels_dict = {}
    for idx, row in df.iterrows():
        filename = str(row['id']).strip()
        if filename.endswith('.mp4'):
            filename = Path(filename).stem  # Remove extension
        
        label_val = str(row['class']).strip().lower()
        
        # Map: deceptive -> 1, truthful -> 0
        if 'deceptive' in label_val:
            labels_dict[filename] = 1
        else:
            labels_dict[filename] = 0
    
    print(f"   ✓ Extracted {len(labels_dict)} labels")
    print(f"     Deceptive: {sum(1 for v in labels_dict.values() if v == 1)}")
    print(f"     Truthful: {sum(1 for v in labels_dict.values() if v == 0)}")
else:
    print("   ✗ pandas not available - cannot read CSV")
    exit(1)

# Step 2: Find all video files and create mapping
print(f"\n2. Finding labeled videos in raw dataset")

raw_dir = r"C:\Users\monic\deception_detector\dataset\raw"
all_videos = glob.glob(raw_dir + r"\**\*.mp4", recursive=True)
print(f"   Found {len(all_videos)} total .mp4 files")

# Map original video paths to standard names
video_mapping = {}  # {video_stem: full_path}

for video_path in all_videos:
    stem = Path(video_path).stem
    if stem in labels_dict:
        video_mapping[stem] = video_path
        
print(f"   ✓ Matched {len(video_mapping)} videos with labels")

# Step 3: Organize videos into clean directory
print(f"\n3. Organizing videos into labeled directories")

videos_dir = r"C:\Users\monic\deception_detector\dataset\videos"
os.makedirs(videos_dir, exist_ok=True)

# Create subdirectories for each class
deceptive_dir = os.path.join(videos_dir, "deceptive")
truthful_dir = os.path.join(videos_dir, "truthful")
os.makedirs(deceptive_dir, exist_ok=True)
os.makedirs(truthful_dir, exist_ok=True)

# Copy videos with proper naming
final_labels = {}
deceptive_count = 0
truthful_count = 0

for video_stem in sorted(video_mapping.keys()):
    label = labels_dict[video_stem]
    src_path = video_mapping[video_stem]
    
    if label == 1:  # Deceptive
        deceptive_count += 1
        dst_filename = f"deceptive_{deceptive_count:03d}.mp4"
        dst_path = os.path.join(deceptive_dir, dst_filename)
        video_id = f"deceptive_{deceptive_count:03d}"
    else:  # Truthful
        truthful_count += 1
        dst_filename = f"truthful_{truthful_count:03d}.mp4"
        dst_path = os.path.join(truthful_dir, dst_filename)
        video_id = f"truthful_{truthful_count:03d}"
    
    # Copy file
    try:
        shutil.copy2(src_path, dst_path)
        final_labels[video_id] = label
        
        if (deceptive_count + truthful_count) % 20 == 0:
            print(f"   Organized {deceptive_count + truthful_count} videos...")
    except Exception as e:
        print(f"   ⚠ Error copying {video_stem}: {e}")

print(f"   ✓ Organized {deceptive_count + truthful_count} videos")
print(f"     - Deceptive: {deceptive_count}")
print(f"     - Truthful: {truthful_count}")

# Step 4: Save final labels
print(f"\n4. Saving labels.json")

labels_file = r"C:\Users\monic\deception_detector\dataset\labels.json"
with open(labels_file, 'w') as f:
    json.dump(final_labels, f, indent=2, sort_keys=True)

print(f"   ✓ Saved {len(final_labels)} labels to labels.json")

# Step 5: Create validation report
print(f"\n5. Creating validation report")

validation_report = f"""
DATASET ORGANIZATION REPORT
============================

Total Videos: {len(final_labels)}
- Deceptive (class 1): {deceptive_count}
- Truthful (class 0): {truthful_count}

Class Balance: {deceptive_count}/{truthful_count} ({'BALANCED' if abs(deceptive_count - truthful_count) <= 1 else 'IMBALANCED'})

Directory Structure:
  dataset/
    ├── videos/
    │   ├── deceptive/     ({deceptive_count} videos)
    │   └── truthful/      ({truthful_count} videos)
    ├── labels.json        ({len(final_labels)} entries)
    ├── features/          (to be populated)
    ├── raw/               (original extracted files)
    └── extract_and_scan.py

Data Format:
  - Labels: JSON with format {{"video_id": 0/1}} where 0=truthful, 1=deceptive
  - Videos: MP4 format in class subdirectories
  - Ready for feature extraction and model training

Next Steps:
  1. Extract audio and text features: python train.py --preprocess
  2. Train model: python train.py --epochs 50
  3. Launch dashboard: streamlit run dashboard.py
"""

report_path = os.path.join(r"C:\Users\monic\deception_detector\dataset", "DATASET_REPORT.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(validation_report)

print(validation_report)

print("\n" + "=" * 70)
print("✅ DATASET ORGANIZATION COMPLETE")
print("=" * 70)
