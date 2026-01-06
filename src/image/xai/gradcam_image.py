from __future__ import annotations
import numpy as np
import torch
import matplotlib.pyplot as plt

def _device(): # Tourner sur gpu si posible pour être plus rapide
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _get_target_layer(model, model_arch: str):
    if "alexnet" in model_arch:
        return model.features[12]
    if "densenet" in model_arch:
        return model.features.denseblock4
    # fallback: essayer dernier module conv trouvé
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m
    raise ValueError("No suitable conv layer found for Grad-CAM.")

def explain_gradcam_torch(model, x_tensor: torch.Tensor, class_idx: int, model_arch: str):
    model.eval()
    x = x_tensor.to(_device())

    target_layer = _get_target_layer(model, model_arch)

    activations = None
    gradients = None

    def fwd_hook(_, __, output):
        nonlocal activations
        activations = output

    def bwd_hook(_, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    logits = model(x)
    score = logits[0, 0] if class_idx == 1 else -logits[0, 0]
    score.backward()

    h1.remove()
    h2.remove()
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)

    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    img = x_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(cam, alpha=0.4)
    plt.axis("off")
    plt.title("Grad-CAM")
    return fig