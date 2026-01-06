from __future__ import annotations
import numpy as np
import shap
import matplotlib.pyplot as plt

from skimage.segmentation import slic


def _make_background(x: np.ndarray, n: int = 12, noise: float = 0.03):
    rng = np.random.default_rng(42)
    bg = np.repeat(x, repeats=n, axis=0)
    if noise > 0:
        bg = bg + rng.normal(0, noise, size=bg.shape).astype(bg.dtype)
        bg = np.clip(bg, 0.0, 1.0)
    return bg


def _shap_gradient_try(model, x_tensor: np.ndarray, class_idx: int, background_size: int, background_noise: float):
    import tensorflow as tf  # On import en local pour eviter le probleme de compatibilité

    background = _make_background(x_tensor, n=background_size, noise=background_noise)

    out = model.output
    if out.shape.rank == 2 and out.shape[-1] == 1:
        target_out = out[:, 0]
    elif out.shape.rank == 2 and out.shape[-1] is not None:
        target_out = out[:, int(class_idx)]
    else:
        target_out = tf.reshape(out, (-1,))

    target_model = tf.keras.Model(inputs=model.inputs, outputs=target_out)

    explainer = shap.GradientExplainer(target_model, background)
    shap_values = explainer.shap_values(x_tensor)
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values
    sv = np.array(sv)
    sv_map = np.mean(sv[0], axis=-1)
    return sv_map


def _mask_image(zs, segmentation, image, background=None):
    if background is None:
        background = image.mean((0, 1))
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]), dtype=image.dtype)
    for i in range(zs.shape[0]):
        out[i] = image
        for j in range(zs.shape[1]):
            if zs[i, j] == 0:
                out[i][segmentation == j] = background
    return out


def _shap_kernel(model, x_tensor: np.ndarray, class_idx: int, n_segments: int = 50, nsamples: int = 200):
    img = x_tensor[0]
    segments = slic(img, n_segments=n_segments, compactness=30, sigma=2, start_label=0)
    n_feats = int(segments.max() + 1)

    def f(z):
        masked = _mask_image(z, segments, img, background=1.0)
        preds = model.predict(masked, verbose=0)
        preds = np.array(preds)
        if preds.ndim == 2 and preds.shape[-1] == 1:
            p_fake = preds[:, 0]
            p_real = 1.0 - p_fake
            preds2 = np.stack([p_real, p_fake], axis=1)
            return preds2[:, class_idx]
        else:
            # (N,2) etc
            return preds[:, class_idx]
    explainer = shap.KernelExplainer(f, np.zeros((1, n_feats)))
    shap_values = explainer.shap_values(np.ones((1, n_feats)), nsamples=nsamples)

    if isinstance(shap_values, list):
        sv = shap_values[0]
    else:
        sv = shap_values

    sv = np.array(sv)[0]
    sv_map = np.zeros(segments.shape, dtype=np.float32)
    for seg_id in range(n_feats):
        sv_map[segments == seg_id] = sv[seg_id]
    return sv_map


def explain_shap(
    model,
    x_tensor: np.ndarray,
    class_idx: int,
    prefer_gradient: bool = True,
    # Gradient params
    background_size: int = 12,
    background_noise: float = 0.03,
    # Kernel params
    n_segments: int = 50,
    nsamples: int = 200,
):
    method_used = None
    try:
        if prefer_gradient:
            sv_map = _shap_gradient_try(model, x_tensor, class_idx, background_size, background_noise)
            method_used = "SHAP GradientExplainer"
        else:
            raise RuntimeError("skip gradient")
    except Exception:
        sv_map = _shap_kernel(model, x_tensor, class_idx, n_segments=n_segments, nsamples=nsamples)
        method_used = "SHAP KernelExplainer (fallback)"

    # On normaliise pour l'overlay
    max_abs = np.max(np.abs(sv_map)) + 1e-8
    sv_norm = sv_map / max_abs

    img = x_tensor[0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].imshow(img)
    ax[0].set_title("Input spectrogram")
    ax[0].axis("off")

    ax[1].imshow(img, alpha=0.35)
    im = ax[1].imshow(sv_norm, cmap="seismic", vmin=-1, vmax=1, alpha=0.85)
    ax[1].set_title(method_used)
    ax[1].axis("off")

    cbar = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    cbar.set_label("SHAP value (normalized)")

    plt.tight_layout()
    return fig