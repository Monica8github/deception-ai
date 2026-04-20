"""
Extract and scan dataset ZIP file
Extracts archive (1).zip, analyzes structure, creates labels.json, and organizes videos
"""

import zipfile
import os
import glob
import shutil
import json
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# STEP 1 - Extract ZIP
print("=" * 70)
print("STEP 1: EXTRACTING ZIP FILE")
print("=" * 70)

zip_path = r"C:\Users\monic\Downloads\archive (1).zip"
extract_path = r"C:\Users\monic\deception_detector\dataset\raw"
os.makedirs(extract_path, exist_ok=True)

try:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        print(f"✓ Total files in ZIP: {len(zf.namelist())}")
        print("\nFirst 20 files:")
        for name in zf.namelist()[:20]:
            print(f"  {name}")
        
        print("\nExtracting...")
        zf.extractall(extract_path)
        print(f"✓ Extraction complete to {extract_path}!")
except Exception as e:
    print(f"✗ Error extracting ZIP: {e}")
    exit(1)

# STEP 2 - Scan and print folder structure
print("\n" + "=" * 70)
print("STEP 2: SCANNING DATASET STRUCTURE")
print("=" * 70)

raw = r"C:\Users\monic\deception_detector\dataset\raw"

videos = (glob.glob(raw + r"\**\*.mp4", recursive=True) +
          glob.glob(raw + r"\**\*.avi", recursive=True) +
          glob.glob(raw + r"\**\*.mov", recursive=True) +
          glob.glob(raw + r"\**\*.mkv", recursive=True))

csvs   = glob.glob(raw + r"\**\*.csv", recursive=True)
jsons  = glob.glob(raw + r"\**\*.json", recursive=True)
txts   = glob.glob(raw + r"\**\*.txt", recursive=True)

print(f"\n✓ Videos found : {len(videos)}")
print(f"✓ CSV files    : {len(csvs)}")
print(f"✓ JSON files   : {len(jsons)}")
print(f"✓ TXT files    : {len(txts)}")

print("\nSample video paths (first 10):")
for i, v in enumerate(videos[:10]):
    print(f"  {i+1}. {v}")

if len(videos) > 10:
    print(f"  ... and {len(videos) - 10} more videos")

print("\nCSV/label files found:")
for c in csvs:
    print(f"  {c}")

# STEP 3 - Read and print first 5 rows of each CSV
print("\n" + "=" * 70)
print("STEP 3: READING CSV FILES")
print("=" * 70)

labels_data = {}

if HAS_PANDAS:
    for csv_file in csvs:
        print(f"\n📋 {Path(csv_file).name}:")
        try:
            df = pd.read_csv(csv_file)
            print(f"Columns: {df.columns.tolist()}")
            print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print("\nFirst 5 rows:")
            print(df.head().to_string())
            
            # Identify label and filename columns
            label_col = None
            filename_col = None
            
            for col in df.columns:
                if any(k in col.lower() for k in ['class', 'label', 'verdict']):
                    label_col = col
                if any(k in col.lower() for k in ['id', 'file', 'name', 'video']):
                    filename_col = col
            
            print(f"\nDetected columns: filename='{filename_col}', label='{label_col}'")
            
            if filename_col and label_col:
                extracted_count = 0
                for idx, row in df.iterrows():
                    try:
                        fname = str(row[filename_col]).strip()
                        if fname.endswith('.mp4'):
                            fname = Path(fname).stem
                        
                        label_val = str(row[label_col]).strip().lower()
                        
                        # Map to 0/1
                        if any(k in label_val for k in ['deceptive', 'lie', 'false']):
                            labels_data[fname] = 1
                            extracted_count += 1
                        elif any(k in label_val for k in ['truthful', 'truth', 'true', 'real']):
                            labels_data[fname] = 0
                            extracted_count += 1
                    except Exception as row_err:
                        print(f"  ⚠ Skipped row {idx}: {row_err}")
                
                print(f"✓ Extracted {extracted_count} labels from CSV")
            else:
                print(f"⚠ Could not identify label columns")
        
        except Exception as e:
            print(f"✗ Error reading {csv_file}: {e}")
else:
    print("⚠ pandas not installed — skipping CSV parsing")
    print("Install with: pip install pandas")

# STEP 4 - Generate labels.json based on filenames if no labels extracted
print("\n" + "=" * 70)
print("STEP 4: GENERATING LABELS.JSON")
print("=" * 70)

# If no labels found in CSVs, infer from filenames and folder structure
if not labels_data:
    print("No labels found in CSV — inferring from filenames and folder structure...")
    
    for video_path in videos:
        video_name = Path(video_path).stem
        parent_folder = Path(video_path).parent.name.lower()
        full_path_str = str(video_path).lower()
        
        # Check filename patterns: trial_lie_XXX or trial_truth_XXX
        if 'lie' in video_name.lower() or 'trial_lie' in video_name.lower():
            labels_data[video_name] = 1
        elif 'truth' in video_name.lower() or 'trial_truth' in video_name.lower():
            labels_data[video_name] = 0
        # Check folder name
        elif any(k in parent_folder for k in ['deceptive', 'lie', 'fake', 'test']):
            if 'lie' in parent_folder or 'deceptive' in parent_folder:
                labels_data[video_name] = 1
            else:
                labels_data[video_name] = 0
else:
    # Also fill in any missing labels by inferring from filenames
    print(f"Found {len(labels_data)} labels from CSV. Inferring remaining {len(videos) - len(labels_data)} from filenames...")
    
    for video_path in videos:
        video_name = Path(video_path).stem
        if video_name not in labels_data:
            # Check filename pattern
            if 'lie' in video_name.lower():
                labels_data[video_name] = 1
            elif 'truth' in video_name.lower():
                labels_data[video_name] = 0

print(f"\nTotal labels generated: {len(labels_data)}")
print(f"  Deceptive (1): {sum(1 for v in labels_data.values() if v == 1)}")
print(f"  Truthful (0):  {sum(1 for v in labels_data.values() if v == 0)}")

# Save labels.json
labels_file = r"C:\Users\monic\deception_detector\dataset\labels.json"
os.makedirs(Path(labels_file).parent, exist_ok=True)

with open(labels_file, 'w') as f:
    json.dump(labels_data, f, indent=2)

print(f"\n✓ Saved labels to {labels_file}")

# STEP 5 - Copy all videos to dataset/videos/
print("\n" + "=" * 70)
print("STEP 5: ORGANIZING VIDEOS")
print("=" * 70)

videos_dir = r"C:\Users\monic\deception_detector\dataset\videos"
os.makedirs(videos_dir, exist_ok=True)

copied_count = 0
for i, v in enumerate(videos):
    try:
        ext = Path(v).suffix
        dst = os.path.join(videos_dir, f"video_{i+1:04d}{ext}")
        shutil.copy2(v, dst)
        copied_count += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Copied {i+1}/{len(videos)} videos...")
    
    except Exception as e:
        print(f"  ✗ Error copying {Path(v).name}: {e}")

print(f"\n✓ Total videos copied: {copied_count}/{len(videos)} to {videos_dir}")

# Update labels.json with new filenames
print("\n" + "=" * 70)
print("STEP 6: UPDATING LABELS WITH NEW FILENAMES")
print("=" * 70)

# Create mapping from old names to new names
old_to_new = {}
for i, v in enumerate(videos):
    old_name = Path(v).stem
    new_name = f"video_{i+1:04d}"
    old_to_new[old_name] = new_name

# Remap labels
new_labels = {}
for old_name, label in labels_data.items():
    if old_name in old_to_new:
        new_labels[old_to_new[old_name]] = label

# Save updated labels.json
with open(labels_file, 'w') as f:
    json.dump(new_labels, f, indent=2)

print(f"✓ Updated labels.json with {len(new_labels)} renamed videos")
print(f"  Final class balance:")
print(f"    Deceptive: {sum(1 for v in new_labels.values() if v == 1)}")
print(f"    Truthful:  {sum(1 for v in new_labels.values() if v == 0)}")

print("\n" + "=" * 70)
print("✅ DATASET EXTRACTION AND ORGANIZATION COMPLETE")
print("=" * 70)
print(f"\nSummary:")
print(f"  • Videos extracted: {len(videos)}")
print(f"  • Videos organized: {copied_count}")
print(f"  • Labels created: {len(new_labels)}")
print(f"\nNext steps:")
print(f"  1. python train.py --preprocess --dataset_dir dataset/")
print(f"  2. python train.py --epochs 50 --batch_size 8")
print(f"  3. streamlit run dashboard.py")
