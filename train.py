import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import config
from models.fusion_model import MultimodalDeceptionDetector
from extractors.video_extractor import VideoExtractor
from extractors.audio_extractor import AudioExtractor
from extractors.text_extractor import TextExtractor


class DeceptionDataset(Dataset):
    """
    Loads pre-extracted features from disk.
    Dataset structure:
    dataset/
      ├── features/
      │   ├── video_001.npy   shape (T, 6)
      │   ├── audio_001.npy   shape (T, 45)
      │   ├── text_001.npy    shape (256,)
      └── labels.json         {"video_001": 1, "video_002": 0, ...}  1=lie, 0=truth
    """

    def __init__(self, dataset_dir: str, split: str = 'train', 
                 test_size: float = 0.2, random_state: int = 42):
        """
        Initialize dataset.
        
        Args:
            dataset_dir: Root directory containing features/ and labels.json
            split: 'train' or 'val'
            test_size: Fraction of data to use for validation
            random_state: Random seed for reproducibility
        """
        self.dataset_dir = Path(dataset_dir)
        self.features_dir = self.dataset_dir / 'features'
        
        # Load labels
        with open(self.dataset_dir / 'labels.json') as f:
            all_labels = json.load(f)
        
        # Split into train/val
        all_keys = list(all_labels.keys())
        train_keys, val_keys = train_test_split(
            all_keys,
            test_size=test_size,
            random_state=random_state,
            stratify=[all_labels[k] for k in all_keys]
        )
        
        if split == 'train':
            self.keys = train_keys
        else:
            self.keys = val_keys
        
        self.labels = {k: all_labels[k] for k in self.keys}

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        """
        Load and return features for one sample.
        
        Returns:
            video_feat: (6,) - mean of video features across frames
            audio_feat: (45,) - mean of audio features across frames
            text_emb: (256,) - text embedding
            label: scalar (0 or 1)
        """
        key = self.keys[idx]
        
        # Load video features (T, 6) → mean → (6,)
        video_path = self.features_dir / f'video_{key}.npy'
        if video_path.exists():
            video_feat = np.load(video_path)  # (T, 6)
            # Pad or truncate to fixed size (64 frames)
            if video_feat.shape[0] < 64:
                pad_size = 64 - video_feat.shape[0]
                video_feat = np.pad(video_feat, ((0, pad_size), (0, 0)), mode='constant')
            else:
                video_feat = video_feat[:64]
            video_feat = np.mean(video_feat, axis=0)  # (6,)
        else:
            video_feat = np.zeros(6)
        
        # Load audio features (T, 45) → mean → (45,)
        audio_path = self.features_dir / f'audio_{key}.npy'
        if audio_path.exists():
            audio_feat = np.load(audio_path)  # (T, 45)
            # Pad or truncate to fixed size
            if audio_feat.shape[0] < 64:
                pad_size = 64 - audio_feat.shape[0]
                audio_feat = np.pad(audio_feat, ((0, pad_size), (0, 0)), mode='constant')
            else:
                audio_feat = audio_feat[:64]
            audio_feat = np.mean(audio_feat, axis=0)  # (45,)
        else:
            audio_feat = np.zeros(45)
        
        # Load text embedding
        text_path = self.features_dir / f'text_{key}.npy'
        if text_path.exists():
            text_emb = np.load(text_path)  # (256,)
        else:
            text_emb = np.zeros(config.EMBED_DIM)
        
        # Load label
        label = self.labels[key]
        
        return (
            torch.FloatTensor(video_feat),
            torch.FloatTensor(audio_feat),
            torch.FloatTensor(text_emb),
            torch.LongTensor([label]).squeeze()
        )


class FeatureExtractorPipeline:
    """
    Pre-extracts and saves features from raw videos to disk.
    Run ONCE before training to avoid re-extracting every epoch.
    """

    def __init__(self):
        """Initialize extractors."""
        self.video_ext = VideoExtractor()
        self.audio_ext = AudioExtractor()
        self.text_ext = TextExtractor()

    def extract_and_save(self, video_path: str, save_dir: str, sample_id: str):
        """
        Extract features from one video and save to disk.
        
        Args:
            video_path: Path to video file
            save_dir: Directory to save features
            sample_id: Sample identifier (e.g., "deceptive_001")
        """
        save_path = Path(save_dir) / 'features'
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Extract video features
        try:
            video_features, _ = self.video_ext.extract(video_path)
            if video_features is not None and video_features.size > 0:
                video_npy = save_path / f'video_{sample_id}.npy'
                np.save(video_npy, video_features)
                print(f"  ✅ Video features: {video_features.shape}")
            else:
                raise ValueError("Empty video features")
        except Exception as e:
            print(f"  ⚠ Video extraction failed: {e}")
            np.save(save_path / f'video_{sample_id}.npy', np.zeros((100, 6)))
        
        # Extract audio features
        try:
            audio_features, _ = self.audio_ext.extract(video_path)
            if audio_features is not None and audio_features.size > 0:
                audio_npy = save_path / f'audio_{sample_id}.npy'
                np.save(audio_npy, audio_features)
                print(f"  ✅ Audio features: {audio_features.shape}")
            else:
                raise ValueError("Empty audio features")
        except Exception as e:
            print(f"  ⚠ Audio extraction failed: {e}")
            np.save(save_path / f'audio_{sample_id}.npy', np.zeros((100, 45)))
        
        # Extract text embedding
        try:
            text_result = self.text_ext.extract(video_path)
            if isinstance(text_result, dict) and 'embedding' in text_result:
                text_emb = text_result['embedding']
            elif isinstance(text_result, np.ndarray):
                text_emb = text_result
            else:
                text_emb = np.zeros(config.EMBED_DIM)
            
            if text_emb is not None and text_emb.size > 0:
                text_npy = save_path / f'text_{sample_id}.npy'
                np.save(text_npy, text_emb)
                print(f"  ✅ Text features : {text_emb.shape}")
            else:
                raise ValueError("Empty text embedding")
        except Exception as e:
            print(f"  ⚠ Text extraction failed: {e}")
            np.save(save_path / f'text_{sample_id}.npy', np.zeros(config.EMBED_DIM))

    def process_dataset(self, video_dir: str, labels: dict, save_dir: str):
        """
        Process entire dataset.
        Handles both flat and subfolder structure (deceptive/, truthful/).
        
        Args:
            video_dir: Directory containing video files or subdirectories
            labels: Dict mapping sample_id → label (0 or 1)
            save_dir: Directory to save extracted features
        """
        video_dir = Path(video_dir)
        processed = 0
        skipped = 0
        failed = []
        
        total_samples = len(labels)
        
        for idx, sample_id in enumerate(sorted(labels.keys()), 1):
            features_path = Path(save_dir) / 'features' / f'video_{sample_id}.npy'
            
            if features_path.exists():
                skipped += 1
                continue
            
            # Find video file - check subfolder structure first
            video_path = None
            
            # Try subdirectories (deceptive/, truthful/)
            for class_dir in ['deceptive', 'truthful']:
                for ext in ['mp4', 'avi', 'mov', 'mkv']:
                    candidate = video_dir / class_dir / f'{sample_id}.{ext}'
                    if candidate.exists():
                        video_path = str(candidate)
                        break
                if video_path:
                    break
            
            # Fall back to flat directory
            if video_path is None:
                for ext in ['mp4', 'avi', 'mov', 'mkv']:
                    candidate = video_dir / f'{sample_id}.{ext}'
                    if candidate.exists():
                        video_path = str(candidate)
                        break
            
            if video_path is None:
                print(f"[{idx:03d}/{total_samples}] ⚠ Video not found: {sample_id}")
                failed.append(sample_id)
                continue
            
            print(f"[{idx:03d}/{total_samples}] Extracting features from {sample_id}.mp4...")
            try:
                self.extract_and_save(video_path, save_dir, sample_id)
                processed += 1
            except Exception as e:
                print(f"[{idx:03d}/{total_samples}] ✗ Failed: {e}")
                failed.append(sample_id)
        
        print(f"\n{'='*70}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"Processed: {processed}/{total_samples}")
        print(f"Skipped (already extracted): {skipped}")
        if failed:
            print(f"Failed: {len(failed)} videos")
            with open(Path(save_dir) / 'failed_extractions.txt', 'w') as f:
                for vid in failed:
                    f.write(f"{vid}\n")
            print(f"Failed videos saved to: {Path(save_dir)}/failed_extractions.txt")


# Training configuration - Optimized for small dataset (121 videos)
TRAINING_CONFIG = {
    'batch_size': 8,                # Reduced for small dataset
    'epochs': 50,
    'learning_rate': 1e-3,          # Higher LR for small dataset
    'weight_decay': 1e-3,           # Stronger L2 regularization
    'patience': 15,                 # Longer patience for small dataset
    'min_delta': 0.001,
    'grad_clip': 1.0,
    'label_smoothing': 0.1,
    'warmup_epochs': 3,             # Shorter warmup
    'checkpoint_dir': 'checkpoints/',
    'log_dir': 'logs/',
    'use_class_weights': True,      # Use inverse frequency weighting
    'use_mixup': True,              # MixUp augmentation
    'mixup_alpha': 0.4,             # Stronger MixUp
    'use_scheduler': 'reducelr',    # ReduceLROnPlateau instead of Cosine
}


class Trainer:
    """Training loop manager with validation and early stopping."""

    def __init__(self, model, train_loader, val_loader, config):
        """
        Initialize trainer.
        
        Args:
            model: MultimodalDeceptionDetector instance
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: Training configuration dict
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.DEVICE if hasattr(config, 'DEVICE') else 'cpu'

        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config['label_smoothing']
        )

        # Optimizer - only train encoders and classifier, freeze DistilBERT body
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # Learning rate scheduler - ReduceLROnPlateau for small datasets
        if config.get('use_scheduler') == 'reducelr':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
            self.use_reducelr = True
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs'] - config['warmup_epochs']
            )
            self.use_reducelr = False

        # Tracking
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1': [],
            'val_auc': []
        }

        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)

    def _warmup_lr(self, epoch: int):
        """Linear warmup for first warmup_epochs."""
        if epoch < self.config['warmup_epochs']:
            lr = self.config['learning_rate'] * (epoch + 1) / self.config['warmup_epochs']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def train_epoch(self, epoch: int) -> dict:
        """
        One training epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dict with train loss and accuracy
        """
        self.model.train()
        self._warmup_lr(epoch)
        
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_idx, batch in enumerate(self.train_loader):
            video_feat, audio_feat, text_emb, labels = batch
            
            video_feat = video_feat.to(self.device)
            audio_feat = audio_feat.to(self.device)
            text_emb = text_emb.to(self.device)
            labels = labels.to(self.device)

            # Encode features
            with torch.no_grad():
                # Video: (B, 6) → (B, 256) through visual encoder
                video_emb = self.model.visual_encoder(video_feat)
                
                # Audio: (B, 45) → (B, 256) through audio encoder
                audio_emb = self.model.audio_encoder(audio_feat)
                
                # Text is already (B, 256)
                text_emb_proc = text_emb

            # Forward pass through fusion and classifier
            logits, _ = self.model(video_emb, audio_emb, text_emb_proc)

            # Compute loss
            loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['grad_clip']
            )
            
            self.optimizer.step()

            # Tracking
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().detach().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().detach().numpy())

            if (batch_idx + 1) % max(10, len(self.train_loader) // 5) == 0:
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return {'loss': avg_loss, 'accuracy': accuracy}

    def validate(self) -> dict:
        """
        Validation epoch.
        
        Returns:
            Dict with val loss, accuracy, F1, and AUC
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                video_feat, audio_feat, text_emb, labels = batch
                
                video_feat = video_feat.to(self.device)
                audio_feat = audio_feat.to(self.device)
                text_emb = text_emb.to(self.device)
                labels = labels.to(self.device)

                # Encode features
                video_emb = self.model.visual_encoder(video_feat)
                audio_emb = self.model.audio_encoder(audio_feat)
                text_emb_proc = text_emb

                # Forward pass
                logits, _ = self.model(video_emb, audio_emb, text_emb_proc)

                # Loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                # Predictions
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().detach().numpy()
                all_preds.extend(preds)
                all_probs.extend(probs[:, 1].cpu().detach().numpy())
                all_labels.extend(labels.cpu().detach().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc
        }

    def fit(self):
        """Full training loop with early stopping and checkpointing."""
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)

        for epoch in range(self.config['epochs']):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate()
            
            # Learning rate schedule (handled after validation now)
            
            # Store metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_auc'].append(val_metrics['auc'])

            # Print epoch summary
            print(
                f"Epoch {epoch+1:3d}/{self.config['epochs']} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.1%} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.1%} "
                f"F1: {val_metrics['f1']:.3f} AUC: {val_metrics['auc']:.3f}"
            )

            # Update scheduler if using ReduceLROnPlateau
            if self.use_reducelr:
                self.scheduler.step(val_metrics['f1'])
            else:
                if epoch >= self.config['warmup_epochs']:
                    self.scheduler.step()
            
            # Early stopping + checkpointing
            if val_metrics['f1'] > self.best_val_f1 + self.config['min_delta']:
                self.best_val_f1 = val_metrics['f1']
                self.patience_counter = 0
                
                # Save best model
                best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pt')
                torch.save(self.model.state_dict(), best_path)
                print(f"  → Saved best model (F1: {self.best_val_f1:.3f})")
            else:
                self.patience_counter += 1
                if self.patience_counter % 5 == 0:
                    print(f"  → No improvement for {self.patience_counter} epochs")
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"\n⚠ Early stopping (no improvement for {self.config['patience']} epochs)")
                break

        # Save training history
        history_path = os.path.join(self.config['log_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\n✓ Training history saved to {history_path}")

        # Plot and save curves
        self._plot_training_curves()

    def _plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor('#0a0a0f')

        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train', color='#e63946')
        axes[0, 0].plot(self.history['val_loss'], label='Val', color='#2ec4b6')
        axes[0, 0].set_title('Loss', color='white')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.2)
        axes[0, 0].set_facecolor('#12121a')

        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train', color='#e63946')
        axes[0, 1].plot(self.history['val_acc'], label='Val', color='#2ec4b6')
        axes[0, 1].set_title('Accuracy', color='white')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.2)
        axes[0, 1].set_facecolor('#12121a')

        # F1 Score
        axes[1, 0].plot(self.history['val_f1'], label='Val F1', color='#f4a261')
        axes[1, 0].set_title('F1 Score', color='white')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.2)
        axes[1, 0].set_facecolor('#12121a')

        # AUC
        axes[1, 1].plot(self.history['val_auc'], label='Val AUC', color='#2ec4b6')
        axes[1, 1].set_title('AUC-ROC', color='white')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.2)
        axes[1, 1].set_facecolor('#12121a')

        # Style
        for ax in axes.flat:
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#1e1e2e')

        plt.tight_layout()
        curves_path = os.path.join(self.config['log_dir'], 'training_curves.png')
        plt.savefig(curves_path, facecolor='#0a0a0f', dpi=100)
        print(f"✓ Training curves saved to {curves_path}")
        plt.close()


class AccuracyBooster:
    """Techniques to improve model accuracy."""

    @staticmethod
    def get_class_weights(labels: list) -> torch.Tensor:
        """
        Compute inverse frequency class weights.
        
        Args:
            labels: List of class labels
        
        Returns:
            Tensor of class weights on config.DEVICE
        """
        labels = np.array(labels)
        classes = np.unique(labels)
        weights = []
        
        for c in classes:
            count = np.sum(labels == c)
            weight = len(labels) / (len(classes) * count)
            weights.append(weight)
        
        return torch.FloatTensor(weights).to(config.DEVICE)

    @staticmethod
    def mixup_batch(video, audio, text, labels, alpha=0.2):
        """
        MixUp augmentation.
        
        Args:
            video, audio, text: Feature tensors (B, ...)
            labels: Label tensor (B,)
            alpha: Beta distribution parameter
        
        Returns:
            Mixed features and (labels_a, labels_b, lambda)
        """
        lam = np.random.beta(alpha, alpha)
        batch_size = video.size(0)
        index = torch.randperm(batch_size)

        mixed_video = lam * video + (1 - lam) * video[index]
        mixed_audio = lam * audio + (1 - lam) * audio[index]
        mixed_text = lam * text + (1 - lam) * text[index]

        return (
            mixed_video, mixed_audio, mixed_text,
            labels, labels[index], lam
        )

    @staticmethod
    def mixup_loss(criterion, pred, labels_a, labels_b, lam):
        """
        MixUp loss computation.
        
        Args:
            criterion: Loss function
            pred: Model predictions (B, 2)
            labels_a, labels_b: Original labels
            lam: MixUp parameter
        
        Returns:
            Mixed loss
        """
        return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)

    @staticmethod
    def add_feature_noise(features: torch.Tensor, std: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise for augmentation."""
        noise = torch.normal(0, std, size=features.shape, device=features.device)
        return features + noise

    @staticmethod
    def get_augmented_video_features(features: np.ndarray) -> np.ndarray:
        """
        Augment video features (temporal crop + noise).
        
        Args:
            features: (T, 6) video features
        
        Returns:
            Augmented features same shape
        """
        T = features.shape[0]
        
        # Random temporal crop (use 80% of frames)
        crop_size = int(T * 0.8)
        start = np.random.randint(0, T - crop_size + 1)
        cropped = features[start:start + crop_size]
        
        # Pad back to original size
        pad_size = T - crop_size
        augmented = np.pad(cropped, ((0, pad_size), (0, 0)), mode='constant')
        
        # Add noise
        augmented = augmented + np.random.normal(0, 0.005, augmented.shape)
        
        # Random horizontal flip of head pose (negate yaw, index 4)
        if np.random.rand() > 0.5:
            augmented[:, 4] *= -1
        
        return augmented


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train MultimodalDeceptionDetector')
    parser.add_argument('--dataset_dir', default='dataset/',
                        help='Root directory with features/ and labels.json')
    parser.add_argument('--video_dir', default='dataset/videos/',
                        help='Directory containing video files')
    parser.add_argument('--labels_file', default='dataset/labels.json',
                        help='Path to labels.json file')
    parser.add_argument('--preprocess', action='store_true',
                        help='Run feature extraction before training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Feature extraction (optional)
    if args.preprocess:
        print("\n" + "=" * 80)
        print("PRE-EXTRACTING FEATURES")
        print("=" * 80)
        
        if not os.path.exists(args.labels_file):
            print(f"Error: {args.labels_file} not found")
            exit(1)
        
        with open(args.labels_file) as f:
            labels = json.load(f)
        
        pipeline = FeatureExtractorPipeline()
        pipeline.process_dataset(args.video_dir, labels, args.dataset_dir)

    # Load dataset
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    
    train_ds = DeceptionDataset(args.dataset_dir, split='train')
    val_ds = DeceptionDataset(args.dataset_dir, split='val')
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Initialize model
    print("\n" + "=" * 80)
    print("INITIALIZING MODEL")
    print("=" * 80)
    
    model = MultimodalDeceptionDetector()
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=config.DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Resumed from {args.resume}")

    TRAINING_CONFIG['epochs'] = args.epochs
    TRAINING_CONFIG['batch_size'] = args.batch_size

    # Train
    trainer = Trainer(model, train_loader, val_loader, TRAINING_CONFIG)
    trainer.fit()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"✓ Best model saved to: {TRAINING_CONFIG['checkpoint_dir']}best_model.pt")
    print(f"✓ Training history saved to: {TRAINING_CONFIG['log_dir']}training_history.json")
    print(f"✓ Training curves saved to: {TRAINING_CONFIG['log_dir']}training_curves.png")
