import sys, os
sys.path.insert(0, '.')
os.chdir(r'C:\Users\monic\deception_detector')

import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import config

with open('dataset/labels.json') as f:
    labels = json.load(f)

valid = []
for k, v in labels.items():
    vp = f'dataset/features/video_{k}.npy'
    ap = f'dataset/features/audio_{k}.npy'
    tp = f'dataset/features/text_{k}.npy'
    if all(os.path.exists(p) for p in [vp, ap, tp]):
        valid.append((k, v))

print(f'Valid samples with features: {len(valid)}')

keys = [k for k, v in valid]
lbls = [v for k, v in valid]
tr_idx, vl_idx = train_test_split(
    range(len(valid)), test_size=0.2,
    stratify=lbls, random_state=42)


def load_feat(k):
    def safe(path, dim):
        try:
            a = np.load(path)
            if a.ndim > 1:
                a = a.mean(axis=0)
            a = a[:dim] if len(a) >= dim else np.pad(a, (0, dim-len(a)))
            s = a.std()
            if s > 1e-8:
                a = (a-a.mean())/s
            return a.astype(np.float32)
        except:
            return np.zeros(dim, dtype=np.float32)
    v = safe(f'dataset/features/video_{k}.npy', 6)
    a = safe(f'dataset/features/audio_{k}.npy', 45)
    t = safe(f'dataset/features/text_{k}.npy', 256)
    return v, a, t


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ve = nn.Sequential(
            nn.Linear(6,64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64,128), nn.ReLU())
        self.ae = nn.Sequential(
            nn.Linear(45,128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,128), nn.ReLU())
        self.te = nn.Sequential(
            nn.Linear(256,128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.3))
        self.av = nn.Linear(128,1,bias=False)
        self.aa = nn.Linear(128,1,bias=False)
        self.at = nn.Linear(128,1,bias=False)
        self.cls = nn.Sequential(
            nn.Linear(128,64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64,2))

    def forward(self,v,a,t):
        V,A,T = self.ve(v), self.ae(a), self.te(t)
        sc = torch.softmax(
            torch.cat([self.av(V), self.aa(A), self.at(T)], 1), 1)
        F = sc[:,0:1]*V + sc[:,1:2]*A + sc[:,2:3]*T
        return self.cls(F), {
            'video': float(sc[:,0].mean().item()),
            'audio': float(sc[:,1].mean().item()),
            'text': float(sc[:,2].mean().item())}


model = Model().to(config.DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='max', factor=0.5, patience=8)

os.makedirs('checkpoints', exist_ok=True)
best_f1 = 0

print('Training...')
print('='*60)

for epoch in range(80):
    model.train()
    tr_loss, tr_correct, tr_total = 0, 0, 0
    idx_list = list(tr_idx)
    np.random.shuffle(idx_list)

    for i in range(0, len(idx_list), 8):
        batch = idx_list[i:i+8]
        vb, ab, tb, yb = [], [], [], []
        for bi in batch:
            k = keys[bi]
            v,a,t = load_feat(k)
            vb.append(v); ab.append(a); tb.append(t); yb.append(lbls[bi])
        V = torch.FloatTensor(np.array(vb)).to(config.DEVICE)
        A = torch.FloatTensor(np.array(ab)).to(config.DEVICE)
        T = torch.FloatTensor(np.array(tb)).to(config.DEVICE)
        Y = torch.LongTensor(yb).to(config.DEVICE)
        opt.zero_grad()
        logits, _ = model(V,A,T)
        loss = loss_fn(logits, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tr_loss += loss.item()
        tr_correct += (logits.argmax(1)==Y).sum().item()
        tr_total += len(Y)

    model.eval()
    preds, trues, probs = [], [], []
    with torch.no_grad():
        for bi in vl_idx:
            k = keys[bi]
            v,a,t = load_feat(k)
            V = torch.FloatTensor(v).unsqueeze(0).to(config.DEVICE)
            A = torch.FloatTensor(a).unsqueeze(0).to(config.DEVICE)
            T = torch.FloatTensor(t).unsqueeze(0).to(config.DEVICE)
            logits, _ = model(V,A,T)
            p = torch.softmax(logits, 1)[0]
            preds.append(logits.argmax(1).item())
            trues.append(lbls[bi])
            probs.append(p[1].item())

    val_acc = sum(p==t for p,t in zip(preds,trues)) / max(len(trues), 1)
    val_f1 = f1_score(trues, preds, average='macro', zero_division=0)
    try:
        val_auc = roc_auc_score(trues, probs)
    except:
        val_auc = 0.5

    sched.step(val_f1)
    tr_acc = tr_correct / max(tr_total, 1)

    saved = ''
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), 'checkpoints/best_model.pt')
        saved = ' ✅ SAVED'

    if (epoch+1) % 5 == 0 or epoch < 5:
        print(f'Epoch {epoch+1:02d}/80 | '
              f'Loss:{tr_loss/max(max(len(idx_list)//8,1),1):.3f} '
              f'Acc:{tr_acc*100:.1f}% | '
              f'Val Acc:{val_acc*100:.1f}% '
              f'F1:{val_f1:.3f} AUC:{val_auc:.3f}{saved}')

print('='*60)
print(f'Best F1: {best_f1:.4f}')
print('Model saved to checkpoints/best_model.pt')
print('Restart dashboard: streamlit run dashboard.py')
