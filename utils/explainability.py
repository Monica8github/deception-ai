import re
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


def visual_gradcam(
    model: 'torch.nn.Module',
    face_crop: np.ndarray,
    device: str = 'cpu',
) -> np.ndarray:
    """Compute a GradCAM overlay for a face crop image using the CapsuleNetwork.

    If pytorch-grad-cam is available it will be used; otherwise a manual
    gradient-based heatmap is produced. On any failure, the original image
    is returned unchanged.
    """
    try:
        if face_crop is None or face_crop.size == 0:
            return face_crop

        image = face_crop.astype(np.float32) / 255.0
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
        tensor.requires_grad = True

        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image

            target_layer = model.conv1[0]
            cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=False)
            grayscale_cam = cam(input_tensor=tensor)[0]
            heatmap = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
            heatmap = cv2.cvtColor((heatmap * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(face_crop.astype(np.uint8), 0.6, heatmap, 0.4, 0)
            return overlay
        except Exception:
            activation = {'value': None}
            gradients = {'value': None}

            def forward_hook(module, input_, output):
                activation['value'] = output

            def backward_hook(module, grad_input, grad_output):
                gradients['value'] = grad_output[0]

            handle_fwd = model.conv1[0].register_forward_hook(forward_hook)
            handle_bwd = model.conv1[0].register_backward_hook(backward_hook)

            out = model(tensor)
            loss = out.norm()
            loss.backward()

            handle_fwd.remove()
            handle_bwd.remove()

            if activation['value'] is None or gradients['value'] is None:
                return face_crop

            act = activation['value'][0]
            grad = gradients['value'][0]
            weights = grad.mean(dim=(1, 2), keepdim=True)
            cam_map = torch.relu((weights * act).sum(dim=0)).cpu().numpy()
            heatmap = cam_map
            heatmap = cv2.resize(heatmap, (face_crop.shape[1], face_crop.shape[0]))
            heatmap = heatmap - heatmap.min()
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(face_crop.astype(np.uint8), 0.6, heatmap, 0.4, 0)
            return overlay
    except Exception:
        return face_crop


def audio_stress_plot(
    stress_score: np.ndarray,
    threshold_pct: float = 0.85,
) -> 'matplotlib.figure.Figure':
    """Plot Sa(t) stress score over time with peak markers."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(stress_score, linewidth=1.2, color='#E8593C', label='Stress Sa(t)')
    threshold = np.percentile(stress_score, threshold_pct * 100)
    peaks = np.where(stress_score > threshold)[0]
    ax.axhline(threshold, linestyle='--', color='#BA7517', linewidth=0.8, label='Peak threshold')
    if peaks.size > 0:
        ax.scatter(peaks, stress_score[peaks], color='#BA7517', s=12, zorder=5, label='Stress peak')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Sa(t)')
    ax.set_title('Vocal Stress Score Over Time')
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def text_attention_highlight(
    tokens: List[str],
    attn_weights: np.ndarray,
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """Return top-n suspicious tokens by normalized attention weight."""
    if attn_weights.size == 0 or len(tokens) == 0:
        return []

    scores = attn_weights.astype(np.float32)
    max_score = float(np.max(scores))
    if max_score > 0:
        norm_scores = scores / max_score
    else:
        norm_scores = scores

    candidates: List[Tuple[str, float]] = []
    for token, score in zip(tokens, norm_scores.tolist()):
        if token in ('[CLS]', '[SEP]'):
            continue
        if token.startswith('##'):
            continue
        if re.match(r'^[^a-zA-Z0-9]+$', token):
            continue
        candidates.append((token, float(score)))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[:top_n]


def fusion_weight_chart(weights: Dict[str, float]) -> 'matplotlib.figure.Figure':
    """Plot modality weights as a bar chart."""
    fig, ax = plt.subplots(figsize=(5, 3))
    labels = ['Video', 'Audio', 'Text']
    values = [weights.get('video', 0.0), weights.get('audio', 0.0), weights.get('text', 0.0)]
    bars = ax.bar(labels, values, color=['#534AB7', '#1D9E75', '#D85A30'], width=0.5)
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Attention weight α')
    ax.set_title('Modality contribution to prediction')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    import numpy as np

    stress = np.random.exponential(0.5, 300)
    fig = audio_stress_plot(stress)
    fig.savefig('stress_test.png', dpi=80)
    plt.close(fig)
    print('Stress plot saved to stress_test.png')

    weights = {'video': 0.45, 'audio': 0.30, 'text': 0.25}
    fig = fusion_weight_chart(weights)
    fig.savefig('fusion_test.png', dpi=80)
    plt.close(fig)
    print('Fusion chart saved to fusion_test.png')

    tokens = ['[CLS]', 'I', 'was', 'at', 'home', 'all', 'evening', ',', 'I', 'swear', '[SEP]']
    attn = np.array([0.1, 0.6, 0.3, 0.9, 0.4, 0.2, 0.8, 0.05, 0.55, 0.95, 0.1])
    top5 = text_attention_highlight(tokens, attn, top_n=5)
    print(f'Top suspicious words: {top5}')
