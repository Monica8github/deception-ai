import glob
import json

ckpt = glob.glob('checkpoints/*.pt')
print('Checkpoints found:', ckpt)

logs = glob.glob('logs/*.json')
print('Log files found:', logs)

features = glob.glob('dataset/features/*.npy')
print(f'Feature files: {len(features)}')

for log in logs:
    with open(log) as f:
        history = json.load(f)
    if history.get('val_f1'):
        epochs_done = len(history['val_f1'])
        best_f1 = max(history['val_f1'])
        best_acc = max(history.get('val_acc', [0]))
        print(f'Epochs completed: {epochs_done}')
        print(f'Best F1 so far  : {best_f1:.4f}')
        print(f'Best Accuracy   : {best_acc:.4f}')
