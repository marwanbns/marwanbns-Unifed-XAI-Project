from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from lime import lime_image


def explain_lime_spectrogram(
    image_rgb_float: np.ndarray,
    predict_fn,
    class_idx: int,
    num_samples: int = 1000,
    num_features: int = 8,
): # Methode pour le spectrogramme lime

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_rgb_float.astype("float64"),
        predict_fn,
        hide_color=0,
        num_samples=num_samples,
    )

    temp, mask = explanation.get_image_and_mask(
        class_idx,
        positive_only=False,
        num_features=num_features,
        hide_rest=True,
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(image_rgb_float)
    ax[0].set_title("Input spectrogram")
    ax[0].axis("off")

    ax[1].imshow(mark_boundaries(temp, mask))
    ax[1].set_title("LIME explanation")
    ax[1].axis("off")

    plt.tight_layout()
    return fig