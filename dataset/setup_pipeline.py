"""
Deception Detector Dataset Setup Pipeline
Handles: extraction, analysis, label generation, video organization, 
validation, feature extraction, and training.
"""

import os
import sys
import json
import glob
import zipfile
import shutil
import subprocess
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARNING] cv2 not available — skipping video validation")


class DatasetPipeline:
    """Complete dataset setup and training pipeline."""
    
    def __init__(self):
        self.downloads_dir = r"C:\Users\monic\Downloads"
        self.project_dir = r"C:\Users\monic\deception_detector"
        self.dataset_dir = os.path.join(self.project_dir, "dataset")
        self.raw_dir = os.path.join(self.dataset_dir, "raw")
        self.videos_dir = os.path.join(self.dataset_dir, "videos")
        self.features_dir = os.path.join(self.dataset_dir, "features")
        self.labels_file = os.path.join(self.dataset_dir, "labels.json")
        self.validation_report = os.path.join(self.dataset_dir, "validation_report.txt")
        
        self.stats = {}
        self.labels = {}
        
    def log(self, message, status="INFO"):
        """Print formatted log message."""
        icons = {
            "INFO": "ℹ",
            "SUCCESS": "✅",
            "WARNING": "⚠",
            "ERROR": "❌",
            "PROGRESS": "⏳"
        }
        icon = icons.get(status, "•")
        print(f"{icon} [{status}] {message}")
    
    def step_1_extract_zip(self):
        """Find and extract ZIP file."""
        self.log("Starting Step 1: Extract ZIP", "PROGRESS")
        
        try:
            # Find ZIP files
            zips = glob.glob(os.path.join(self.downloads_dir, "*.zip"))
            
            # Filter by keywords
            keywords = ["deception", "lie", "truth", "trial", "dataset"]
            relevant_zips = [z for z in zips if any(k.lower() in Path(z).stem.lower() for k in keywords)]
            
            if not relevant_zips and zips:
                relevant_zips = zips  # If no matching keywords, use all zips
            
            if not relevant_zips:
                self.log(f"No ZIP files found in {self.downloads_dir}", "WARNING")
                return False
            
            zip_file = relevant_zips[0]
            self.log(f"Found ZIP: {Path(zip_file).name}")
            
            # Create raw directory
            os.makedirs(self.raw_dir, exist_ok=True)
            
            # Extract
            self.log(f"Extracting {Path(zip_file).name}...")
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(self.raw_dir)
                
                # Show first 30 files
                file_list = sorted(zf.namelist())[:30]
                for fname in file_list:
                    print(f"  {fname}")
                
                if len(zf.namelist()) > 30:
                    print(f"  ... and {len(zf.namelist()) - 30} more files")
            
            self.log(f"Extracted {len(zf.namelist())} files to {self.raw_dir}", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Failed to extract ZIP: {e}", "ERROR")
            return False
    
    def step_2_analyze_dataset(self):
        """Analyze dataset structure."""
        self.log("Starting Step 2: Analyze Dataset", "PROGRESS")
        
        try:
            self.stats = {
                'videos': [],
                'audio': [],
                'text': [],
                'label_files': [],
                'subfolders': []
            }
            
            # Walk through raw directory
            for root, dirs, files in os.walk(self.raw_dir):
                for file in files:
                    # Count videos
                    if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        self.stats['videos'].append(os.path.join(root, file))
                    # Count audio
                    elif file.lower().endswith(('.wav', '.mp3', '.flac')):
                        self.stats['audio'].append(os.path.join(root, file))
                    # Count text
                    elif file.lower().endswith(('.txt', '.csv', '.json')):
                        if any(keyword in file.lower() for keyword in ['label', 'annotation', 'metadata', 'truth', 'ground']):
                            self.stats['label_files'].append(os.path.join(root, file))
                        else:
                            self.stats['text'].append(os.path.join(root, file))
                
                # Track subfolders
                if dirs:
                    self.stats['subfolders'].extend([os.path.basename(d) for d in [os.path.join(root, d) for d in dirs]])
            
            # Print summary
            print("\n" + "=" * 50)
            print("DATASET ANALYSIS SUMMARY")
            print("=" * 50)
            print(f"Total videos     : {len(self.stats['videos'])}")
            print(f"Total audio files: {len(self.stats['audio'])}")
            print(f"Total text files : {len(self.stats['text'])}")
            print(f"Label files found: {len(self.stats['label_files'])}")
            if self.stats['label_files']:
                for lf in self.stats['label_files']:
                    print(f"  - {os.path.relpath(lf, self.raw_dir)}")
            if self.stats['subfolders']:
                print(f"Subfolders      : {', '.join(set(self.stats['subfolders']))}")
            print("=" * 50 + "\n")
            
            self.log(f"Analysis complete: {len(self.stats['videos'])} videos found", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Failed to analyze dataset: {e}", "ERROR")
            return False
    
    def step_3_generate_labels(self):
        """Auto-generate labels.json."""
        self.log("Starting Step 3: Generate Labels", "PROGRESS")
        
        try:
            self.labels = {}
            auto_labeled = 0
            unlabeled = 0
            
            # Case A: Load from existing label file
            if self.stats['label_files']:
                self.log("Found label file(s) — parsing...")
                label_file = self.stats['label_files'][0]
                
                if label_file.lower().endswith('.json'):
                    with open(label_file) as f:
                        raw_labels = json.load(f)
                        self._parse_labels_dict(raw_labels)
                        auto_labeled = len(self.labels)
                
                elif label_file.lower().endswith('.csv'):
                    with open(label_file) as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            self._parse_label_row(row)
                            if len(self.labels) > auto_labeled:
                                auto_labeled = len(self.labels)
                
                self.log(f"Loaded {auto_labeled} labels from {Path(label_file).name}", "SUCCESS")
            
            # Case B: Infer from filename and folder structure
            for video_path in self.stats['videos']:
                video_name = Path(video_path).stem
                
                # Already labeled?
                if video_name in self.labels:
                    continue
                
                # Infer from folder name (Case C)
                folder_name = Path(video_path).parent.name.lower()
                if any(k in folder_name for k in ['deceptive', 'lie', 'fake', 'd_']):
                    self.labels[video_name] = 1
                    auto_labeled += 1
                elif any(k in folder_name for k in ['truthful', 'truth', 'real', 't_']):
                    self.labels[video_name] = 0
                    auto_labeled += 1
                # Infer from filename
                elif any(k in video_name.lower() for k in ['deceptive', 'lie', 'fake', 'd_']):
                    self.labels[video_name] = 1
                    auto_labeled += 1
                elif any(k in video_name.lower() for k in ['truthful', 'truth', 'real', 't_']):
                    self.labels[video_name] = 0
                    auto_labeled += 1
                else:
                    unlabeled += 1
            
            # Save labels.json
            os.makedirs(self.dataset_dir, exist_ok=True)
            with open(self.labels_file, 'w') as f:
                json.dump(self.labels, f, indent=2)
            
            self.log(
                f"Created labels.json: {auto_labeled} auto-labeled, {unlabeled} unlabeled",
                "SUCCESS"
            )
            return True
            
        except Exception as e:
            self.log(f"Failed to generate labels: {e}", "ERROR")
            return False
    
    def _parse_label_row(self, row: dict):
        """Parse a label CSV row."""
        # Find filename column
        filename_col = None
        for col in row.keys():
            if any(k in col.lower() for k in ['file', 'name', 'video', 'id']):
                filename_col = col
                break
        if not filename_col:
            filename_col = list(row.keys())[0]
        
        # Find label column
        label_col = None
        for col in row.keys():
            if any(k in col.lower() for k in ['label', 'class', 'truth', 'deceptive', 'verdict', 'annotation']):
                label_col = col
                break
        if not label_col:
            return
        
        filename = Path(row[filename_col]).stem
        label_val = str(row[label_col]).lower()
        
        # Convert to 0/1
        if any(k in label_val for k in ['deceptive', 'lie', '1', 'fake', 'false']):
            self.labels[filename] = 1
        elif any(k in label_val for k in ['truthful', 'truth', '0', 'real', 'true']):
            self.labels[filename] = 0
    
    def _parse_labels_dict(self, data: dict):
        """Parse labels from dictionary."""
        for key, val in data.items():
            key_clean = Path(key).stem
            if isinstance(val, int):
                self.labels[key_clean] = val
            else:
                val_str = str(val).lower()
                if any(k in val_str for k in ['deceptive', 'lie', '1', 'fake', 'false']):
                    self.labels[key_clean] = 1
                else:
                    self.labels[key_clean] = 0
    
    def step_4_organize_videos(self):
        """Organize videos into standard folder structure."""
        self.log("Starting Step 4: Organize Videos", "PROGRESS")
        
        try:
            os.makedirs(self.videos_dir, exist_ok=True)
            
            lie_count = 0
            truth_count = 0
            new_labels = {}
            
            for video_path in self.stats['videos']:
                video_name = Path(video_path).stem
                ext = Path(video_path).suffix
                
                # Get label
                label = self.labels.get(video_name, None)
                if label is None:
                    self.log(f"Skipping {video_name} (no label)", "WARNING")
                    continue
                
                # Generate new filename
                if label == 1:
                    lie_count += 1
                    new_name = f"lie_{lie_count:03d}{ext}"
                else:
                    truth_count += 1
                    new_name = f"truth_{truth_count:03d}{ext}"
                
                # Copy file
                new_path = os.path.join(self.videos_dir, new_name)
                shutil.copy2(video_path, new_path)
                
                # Update labels with new name
                new_name_clean = Path(new_name).stem
                new_labels[new_name_clean] = label
            
            # Save updated labels
            self.labels = new_labels
            with open(self.labels_file, 'w') as f:
                json.dump(self.labels, f, indent=2)
            
            self.log(
                f"Organized {lie_count + truth_count} videos: {lie_count} deceptive, {truth_count} truthful",
                "SUCCESS"
            )
            return True
            
        except Exception as e:
            self.log(f"Failed to organize videos: {e}", "ERROR")
            return False
    
    def step_5_validate_dataset(self):
        """Validate dataset integrity."""
        self.log("Starting Step 5: Validate Dataset", "PROGRESS")
        
        try:
            os.makedirs(self.features_dir, exist_ok=True)
            report = []
            
            report.append("=" * 60)
            report.append("DATASET VALIDATION REPORT")
            report.append("=" * 60)
            
            # Check files exist
            missing = []
            for label_name in self.labels.keys():
                video_files = glob.glob(os.path.join(self.videos_dir, f"{label_name}.*"))
                if not video_files:
                    missing.append(label_name)
            
            report.append(f"\nFiles labeled: {len(self.labels)}")
            report.append(f"Files missing: {len(missing)}")
            if missing:
                report.append("Missing files:")
                for m in missing[:10]:
                    report.append(f"  - {m}")
            
            # Class balance
            deceptive = sum(1 for v in self.labels.values() if v == 1)
            truthful = sum(1 for v in self.labels.values() if v == 0)
            report.append(f"\nClass Balance:")
            report.append(f"  Deceptive: {deceptive} | Truthful: {truthful}")
            
            # Check video integrity
            corrupt_count = 0
            total_duration = 0.0
            
            if CV2_AVAILABLE:
                report.append(f"\nVideo Integrity Check:")
                for label_name in list(self.labels.keys())[:50]:  # Check first 50
                    video_files = glob.glob(os.path.join(self.videos_dir, f"{label_name}.*"))
                    if video_files:
                        try:
                            cap = cv2.VideoCapture(video_files[0])
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            duration = frame_count / fps if fps > 0 else 0
                            total_duration += duration
                            cap.release()
                        except:
                            corrupt_count += 1
                
                report.append(f"  Checked: 50 videos")
                report.append(f"  Corrupt: {corrupt_count}")
                if len(self.labels) > 0:
                    avg_duration = total_duration / min(50, len(self.labels))
                    report.append(f"  Avg duration: {avg_duration:.1f}s")
            else:
                report.append("\nVideo Integrity Check: SKIPPED (cv2 not available)")
            
            report.append("\n" + "=" * 60)
            report_text = "\n".join(report)
            print(report_text)
            
            # Save report
            with open(self.validation_report, 'w') as f:
                f.write(report_text)
            
            self.log(f"Validation report saved to {self.validation_report}", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Failed validation: {e}", "ERROR")
            return False
    
    def step_6_extract_features(self):
        """Run feature extraction."""
        self.log("Starting Step 6: Feature Extraction", "PROGRESS")
        
        try:
            cmd = [
                sys.executable, "train.py",
                "--preprocess",
                "--dataset_dir", self.dataset_dir,
                "--video_dir", self.videos_dir,
                "--labels_file", self.labels_file
            ]
            
            os.chdir(self.project_dir)
            result = subprocess.run(cmd, capture_output=False)
            
            if result.returncode == 0:
                self.log("Feature extraction completed", "SUCCESS")
                return True
            else:
                self.log(f"Feature extraction failed with code {result.returncode}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Failed to run feature extraction: {e}", "ERROR")
            return False
    
    def step_7_train_model(self):
        """Run training."""
        self.log("Starting Step 7: Train Model", "PROGRESS")
        
        try:
            cmd = [
                sys.executable, "train.py",
                "--epochs", "50",
                "--batch_size", "8",
                "--dataset_dir", self.dataset_dir,
                "--labels_file", self.labels_file
            ]
            
            os.chdir(self.project_dir)
            result = subprocess.run(cmd, capture_output=False)
            
            if result.returncode == 0:
                self.log("Training completed", "SUCCESS")
                return True
            else:
                self.log(f"Training failed with code {result.returncode}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Failed to run training: {e}", "ERROR")
            return False
    
    def run(self):
        """Run complete pipeline."""
        print("\n" + "=" * 70)
        print("DECEPTION DETECTOR — DATASET SETUP PIPELINE")
        print("=" * 70 + "\n")
        
        results = {
            '1': self.step_1_extract_zip,
            '2': self.step_2_analyze_dataset,
            '3': self.step_3_generate_labels,
            '4': self.step_4_organize_videos,
            '5': self.step_5_validate_dataset,
            '6': self.step_6_extract_features,
            '7': self.step_7_train_model,
        }
        
        step_names = {
            '1': 'Extraction',
            '2': 'Analysis',
            '3': 'Labels created',
            '4': 'Videos organized',
            '5': 'Validation',
            '6': 'Feature extraction',
            '7': 'Training',
        }
        
        completed = []
        
        for step, func in results.items():
            if func():
                completed.append(step)
            else:
                self.log(f"Stopping at step {step}", "WARNING")
                break
        
        # Print summary
        print("\n" + "=" * 70)
        print("PIPELINE SUMMARY")
        print("=" * 70)
        
        for step in ['1', '2', '3', '4', '5', '6', '7']:
            if step in completed:
                status = "✅ DONE"
            else:
                status = "⏳ PENDING"
            print(f"{status} Step {step} - {step_names[step]}")
        
        print("=" * 70 + "\n")


if __name__ == '__main__':
    pipeline = DatasetPipeline()
    pipeline.run()
