from __future__ import annotations

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt


def _find_last_conv_layer_name(model: tf.keras.Model): # Trouver automatiquement la derniere couche d'un modèle keras
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError(
        "Impossible de trouver une couche Conv2D. "
    )


def explain_gradcam(
    model: tf.keras.Model,
    x_tensor: np.ndarray,
    class_idx: int,
    last_conv_layer: str | None = None,
):

    if last_conv_layer is None:
        last_conv_layer = _find_last_conv_layer_name(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x_tensor)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    img = (x_tensor[0] * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(img)
    ax[0].set_title("Input spectrogram")
    ax[0].axis("off")

    ax[1].imshow(overlay)
    ax[1].set_title(f"Grad-CAM (layer={last_conv_layer})")
    ax[1].axis("off")

    plt.tight_layout()
    return fig