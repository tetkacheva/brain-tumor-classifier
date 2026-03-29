import torch
import numpy as np
from PIL import Image
import cv2


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        target = model.features[-1]
        target.register_forward_hook(self._save_act)
        target.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):
        self.activations = o

    def _save_grad(self, m, gi, go):
        self.gradients = go[0]

    def generate(self, x, class_idx):
        self.model.zero_grad()
        out = self.model(x)
        out[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam).cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def guided_gradcam(model, x, class_idx):
    model.eval()
    gcam = GradCAM(model)
    cam = gcam.generate(x, class_idx)
    return cam, cam


def apply_heatmap(original_pil: Image.Image, cam: np.ndarray) -> Image.Image:
    w, h = original_pil.size
    cam_resized = cv2.resize(cam, (w, h))

    # Apply colormap
    heatmap = cv2.applyColorMap((255 * cam_resized).astype("uint8"), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    original_np = np.array(original_pil.convert("RGB"))
    overlay = cv2.addWeighted(original_np, 0.5, heatmap, 0.5, 0)
    return Image.fromarray(overlay)
