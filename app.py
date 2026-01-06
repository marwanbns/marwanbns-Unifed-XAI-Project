import streamlit as st
from pathlib import Path
from src.common.io import save_uploaded_file, make_cache_path, ensure_dir
from src.common.utils import is_audio_file, is_image_file
from src.common.registry import get_models_for_input, get_xai_for_input
# Ici on importe les methodes pour les audios
from src.audio.preprocess import wav_to_melspec_png, pil_to_tensor
from src.audio.models import load_audio_model, predict_audio, CLASS_NAMES
from src.audio.xai.lime_audio import explain_lime_spectrogram
from src.audio.xai.gradcam_audio import explain_gradcam
from src.audio.xai.shap_audio import explain_shap
# Ici on importe les methodes pour les images
from src.image.preprocess import load_xray_as_tensor
from src.image.models import load_image_model, predict_image, IMAGE_CLASS_NAMES
from src.image.xai.gradcam_image import explain_gradcam_torch

st.set_page_config(page_title="Unified XAI Platform", layout="wide")


def select_model(input_type: str): # Methode pour la selection du modèle
    models = get_models_for_input(input_type)
    options = {m.name: m for m in models}
    chosen_name = st.selectbox("Model", list(options.keys()))
    return options[chosen_name]


def select_xai(input_type: str): # Methode pour la selectiion du type de methode de XAI (Lime | Shap | Grad Cam)
    xais = get_xai_for_input(input_type)
    options = {x.name: x for x in xais}
    chosen_name = st.selectbox("XAI Method", list(options.keys()))
    return options[chosen_name]


def run_audio_pipeline(saved_path: Path, model_spec, xai_spec): # Methode pour la pipeline de la detection de deeep fake audio
    st.write("### Audio input")
    st.audio(saved_path.read_bytes(), format="audio/wav")

    # Partie pour le spectrogramme
    cache_png = make_cache_path(saved_path, "data/cache", ".png")
    spec_img = wav_to_melspec_png(saved_path, cache_png)
    st.image(spec_img, caption="Mel Spectrogram", width=320)

    # Tensor pour le modele
    x_tensor = pil_to_tensor(spec_img)

    # Chargement du modèle avec predition
    model = load_audio_model(model_spec.path)
    label_idx, probs = predict_audio(model, x_tensor)

    st.success(f"Prediction: **{CLASS_NAMES[label_idx]}**")
    st.write("Probabilities:", probs)

    # XAI
    st.write("### Explainability")
    if xai_spec.id == "lime":
        fig = explain_lime_spectrogram(
            image_rgb_float=x_tensor[0],
            predict_fn=model.predict,
            class_idx=label_idx,
        )
        st.pyplot(fig)

    elif xai_spec.id == "gradcam":
        fig = explain_gradcam(
            model=model,
            x_tensor=x_tensor,
            class_idx=label_idx,
            last_conv_layer=None,
        )
        st.pyplot(fig)

    elif xai_spec.id == "shap":
        fig = explain_shap(
            model=model,
            x_tensor=x_tensor,
            class_idx=label_idx,
            prefer_gradient=True,
            nsamples=200,
            n_segments=50
        )
        st.pyplot(fig)
    else:
        st.error("XAI method not supported.")


def run_image_pipeline(saved_path: Path, model_spec, xai_spec): # Pipeline pour la detection du cancer des poumons
    st.write("### Image input")
    st.image(str(saved_path), caption="Uploaded chest X-ray", use_container_width=True)

    # Preprocess du tensor
    x_tensor = load_xray_as_tensor(saved_path)

    # Chargement du modèle
    model = load_image_model(model_spec)
    label_idx, probs = predict_image(model, x_tensor)

    st.success(f"Prediction: **{IMAGE_CLASS_NAMES[label_idx]}**")
    st.write("Probabilities:", probs)

    st.write("### Explainability")
    if xai_spec.id == "gradcam":
        fig = explain_gradcam_torch(
            model=model,
            x_tensor=x_tensor,
            class_idx=label_idx,
            model_arch=model_spec.id,
        )
        st.pyplot(fig)
    elif xai_spec.id == "lime":
        st.warning("Pas de LIME pour la detection de cancer des poumons") # A priori on ne tombe jamais dans ces cas de figure car l'option disparait en fonction de notre choix de fichiers
    elif xai_spec.id == "shap":
        st.warning("Pas de SHAP pour la detection de cancer des poumons") # Pareil
    else:
        st.error("XAI method not supported.")


def run_audio_compare(saved_path: Path, model_spec, xai_specs): # Tab 2 pour l'audio
    st.write("### Audio input")
    st.audio(saved_path.read_bytes(), format="audio/wav")

    # Pour le spectrogram
    cache_png = make_cache_path(saved_path, "data/cache", ".png")
    spec_img = wav_to_melspec_png(saved_path, cache_png)
    st.image(spec_img, caption="Mel Spectrogram", width=320)

    x_tensor = pil_to_tensor(spec_img)

    # Chargement du modèle
    model = load_audio_model(model_spec.path)
    label_idx, probs = predict_audio(model, x_tensor)

    st.success(f"Prediction: **{CLASS_NAMES[label_idx]}**")
    st.write("Probabilities:", probs)

    # XAI les uns à coté des autres
    st.write("### XAI Comparison")
    if not xai_specs:
        st.warning("Select at least one XAI method.")
        return

    cols = st.columns(len(xai_specs))
    for col, xai_spec in zip(cols, xai_specs):
        with col:
            st.markdown(f"#### {xai_spec.name}")
            if xai_spec.id == "lime":
                fig = explain_lime_spectrogram(
                    image_rgb_float=x_tensor[0],
                    predict_fn=model.predict,
                    class_idx=label_idx,
                )
                st.pyplot(fig)

            elif xai_spec.id == "gradcam":
                fig = explain_gradcam(
                    model=model,
                    x_tensor=x_tensor,
                    class_idx=label_idx,
                    last_conv_layer=None,
                )
                st.pyplot(fig)

            elif xai_spec.id == "shap":
                fig = explain_shap(model, x_tensor, label_idx)
                st.pyplot(fig)

            else:
                st.error("Unsupported XAI")


def run_image_compare(saved_path: Path, model_spec, xai_specs): # Tab 2 pour l'image
    st.write("### Image input")
    st.image(str(saved_path), caption="Uploaded chest X-ray", use_container_width=True)

    x_tensor = load_xray_as_tensor(saved_path)

    model = load_image_model(model_spec)
    label_idx, probs = predict_image(model, x_tensor)

    st.success(f"Prediction: **{IMAGE_CLASS_NAMES[label_idx]}**")
    st.write("Probabilities:", probs)

    st.write("### XAI Comparison")
    if not xai_specs:
        st.warning("Select at least one XAI method.")
        return

    cols = st.columns(len(xai_specs))
    for col, xai_spec in zip(cols, xai_specs):
        with col:
            st.markdown(f"#### {xai_spec.name}")
            if xai_spec.id == "gradcam":
                fig = explain_gradcam_torch(
                    model=model,
                    x_tensor=x_tensor,
                    class_idx=label_idx,
                    model_arch=model_spec.id,
                )
                st.pyplot(fig)
            elif xai_spec.id == "lime":
                st.warning("Non lime")
            elif xai_spec.id == "shap":
                st.warning("Non shap")
            else:
                st.error("Unsupported XAI")


def main():
    st.title("Unified Explainable AI Interface")

    # On verifie que les dossiers existent
    ensure_dir("data/uploads")
    ensure_dir("data/cache")
    ensure_dir("data/outputs")

    tab1, tab2 = st.tabs(["Single run", "Comparison"])

    with tab1:
        st.subheader("Single run (1 model + 1 XAI)")

        dataset = st.selectbox(
            "Select dataset",
            ["Deepfake Audio", "Chest X-ray"]
        )

        uploaded = st.file_uploader(
            "Upload audio (.wav) or image (.png/.jpg)",
            type=["wav", "png", "jpg", "jpeg"],
            key="single_uploader",
        )
        if not uploaded:
            st.info("Upload a file to start.")
        else:
            saved_path = save_uploaded_file(uploaded, "data/uploads")
            st.caption(f"Saved to: {saved_path}")
            if is_audio_file(saved_path):
                input_type = "audio"
            elif is_image_file(saved_path):
                input_type = "image"
            else:
                st.error("Unsupported file type.")
                return

            st.write(f"Detected input type: **{input_type}**")
            model_spec = select_model(input_type)
            xai_spec = select_xai(input_type)

            if st.button("Run (single)"):
                if input_type == "audio":
                    run_audio_pipeline(saved_path, model_spec, xai_spec)
                else:
                    run_image_pipeline(saved_path, model_spec, xai_spec)

    with tab2:
        st.subheader("Comparison (same input, multiple XAI)")

        dataset = st.selectbox(
            "Select dataset",
            ["Deepfake Audio", "Chest X-ray"],
            key="compare_dataset",
        )

        uploaded = st.file_uploader(
            "Upload audio (.wav) or image (.png/.jpg)",
            type=["wav", "png", "jpg", "jpeg"],
            key="compare_uploader",
        )
        if not uploaded:
            st.info("Upload a file to start.")
        else:
            saved_path = save_uploaded_file(uploaded, "data/uploads")
            st.caption(f"Saved to: {saved_path}")
            if is_audio_file(saved_path):
                input_type = "audio"
            elif is_image_file(saved_path):
                input_type = "image"
            else:
                st.error("Unsupported file type.")
                return

            st.write(f"Detected input type: **{input_type}**")

            # Selection du modele
            model_spec = select_model(input_type)

            # Selection du type de XAI
            xais = get_xai_for_input(input_type)
            xai_name_to_spec = {x.name: x for x in xais}
            chosen_xais = st.multiselect(
                "Select XAI methods",
                list(xai_name_to_spec.keys()),
                default=list(xai_name_to_spec.keys())[:2],  # 2 par défaut
            )

            if st.button("Run (compare)"):
                if input_type == "audio":
                    run_audio_compare(saved_path, model_spec, [xai_name_to_spec[n] for n in chosen_xais])
                else:
                    run_image_compare(saved_path, model_spec, [xai_name_to_spec[n] for n in chosen_xais])


if __name__ == "__main__":
    main()