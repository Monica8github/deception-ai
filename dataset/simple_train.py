import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import config


class SimpleDeceptionDataset(Dataset):
    def __init__(self, feature_dir, labels, indices):
        self.feature_dir = feature_dir
        self.keys = [list(labels.keys())[i] for i in indices]
        self.labels = labels

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        label = self.labels[key]

        def load(path, target_shape):
            try:
                arr = np.load(path)
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                arr = arr.astype(np.float32)
                if len(arr) < target_shape:
                    arr = np.pad(arr, (0, target_shape - len(arr)), mode='constant')
                elif len(arr) > target_shape:
                    arr = arr[:target_shape]
                return arr
            except Exception:
                return np.zeros(target_shape, dtype=np.float32)

        v = load(f'dataset/features/video_{key}.npy', config.VIDEO_FEAT_DIM)
        a = load(f'dataset/features/audio_{key}.npy', config.AUDIO_FEAT_DIM)
        t = load(f'dataset/features/text_{key}.npy', config.EMBED_DIM)

        for arr in [v, a, t]:
            std = arr.std()
            if std > 0:
                arr -= arr.mean()
                arr /= std

        return (
            torch.FloatTensor(v),
            torch.FloatTensor(a),
            torch.FloatTensor(t),
            torch.LongTensor([label]).squeeze()
        )


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        probs = torch.softmax(inputs, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = self.alpha * ((1 - p_t) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class SimpleDeceptionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.video_fc1 = nn.Sequential(
            nn.Linear(config.VIDEO_FEAT_DIM, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.video_fc2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.audio_fc1 = nn.Sequential(
            nn.Linear(config.AUDIO_FEAT_DIM, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.audio_fc2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.text_fc1 = nn.Sequential(
            nn.Linear(config.EMBED_DIM, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.text_fc2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.cross_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.attn_v = nn.Linear(128, 1, bias=False)
        self.attn_a = nn.Linear(128, 1, bias=False)
        self.attn_t = nn.Linear(128, 1, bias=False)
        self.attn_c = nn.Linear(128, 1, bias=False)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, v, a, t):
        v1 = self.video_fc1(v)
        v2 = self.video_fc2(v1)
        V = v2

        a1 = self.audio_fc1(a)
        a2 = self.audio_fc2(a1)
        A = a2

        t1 = self.text_fc1(t)
        t2 = self.text_fc2(t1)
        T = t2

        cross_raw = torch.cat([V, A], dim=1)
        C = self.cross_fc(cross_raw)

        score = torch.softmax(
            torch.cat([self.attn_v(V), self.attn_a(A), self.attn_t(T), self.attn_c(C)], dim=1),
            dim=1,
        )
        av, aa, at_, ac = score[:, 0:1], score[:, 1:2], score[:, 2:3], score[:, 3:4]
        fused = av * V + aa * A + at_ * T + ac * C

        logits = self.classifier(fused)
        return logits, {
            'video': av.mean().item(),
            'audio': aa.mean().item(),
            'text': at_.mean().item(),
            'cross': ac.mean().item(),
        }


def load_labels():
    with open('dataset/labels.json') as f:
        return json.load(f)


labels = load_labels()
valid_keys = []
for k in labels:
    v_ok = os.path.exists(f'dataset/features/video_{k}.npy')
    a_ok = os.path.exists(f'dataset/features/audio_{k}.npy')
    t_ok = os.path.exists(f'dataset/features/text_{k}.npy')
    if v_ok and a_ok and t_ok:
        valid_keys.append(k)
    else:
        print(f'Missing features for {k}: video={v_ok}, audio={a_ok}, text={t_ok}')

labels = {k: labels[k] for k in valid_keys}
print(f'Valid samples: {len(labels)}')

all_idx = list(range(len(labels)))
label_vals = [list(labels.values())[i] for i in all_idx]
train_idx, val_idx = train_test_split(
    all_idx, test_size=0.2, random_state=42, stratify=label_vals
)

train_ds = SimpleDeceptionDataset('dataset/features', labels, train_idx)
val_ds = SimpleDeceptionDataset('dataset/features', labels, val_idx)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

model = SimpleDeceptionModel().to(config.DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
criterion = FocalLoss(alpha=0.25, gamma=2.0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('logs', exist_ok=True)

best_f1 = 0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': [], 'val_auc': []}

print('Starting training...')
print('=' * 70)

for epoch in range(50):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for v, a, t, y in train_loader:
        v, a, t, y = v.to(config.DEVICE), a.to(config.DEVICE), t.to(config.DEVICE), y.to(config.DEVICE)
        optimizer.zero_grad()
        logits, _ = model(v, a, t)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)

    train_acc = correct / max(total, 1)
    train_loss /= max(len(train_loader), 1)

    model.eval()
    val_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    with torch.no_grad():
        for v, a, t, y in val_loader:
            v, a, t, y = v.to(config.DEVICE), a.to(config.DEVICE), t.to(config.DEVICE), y.to(config.DEVICE)
            logits, _ = model(v, a, t)
            loss = criterion(logits, y)
            val_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_loss /= max(len(val_loader), 1)
    val_acc = sum(int(p == l) for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)
    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    try:
        val_auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        val_auc = 0.5

    scheduler.step(val_f1)

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), 'checkpoints/best_model_v2.pt')
        torch.save(model.state_dict(), 'checkpoints/best_model.pt')
        saved = '[SAVED]'
    else:
        saved = ''

    print(
        f'Epoch {epoch+1:02d}/50 | Train Loss: {train_loss:.3f} Acc: {train_acc*100:.1f}% | '
        f'Val Loss: {val_loss:.3f} Acc: {val_acc*100:.1f}% F1: {val_f1:.3f} AUC: {val_auc:.3f} {saved}'
    )

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    history['val_auc'].append(val_auc)

    with open('logs/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open('logs/training_history_v2.json', 'w') as f:
        json.dump(history, f, indent=2)

print('=' * 70)
print(f'Training complete! Best F1: {best_f1:.4f}')
print('Model saved to: checkpoints/best_model_v2.pt')
