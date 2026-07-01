import os

import streamlit as st

from tabs.tasks.utils import browse_file, browse_folder


IMAGE_SEGMENTATION_DATASETS = [
    "Cityscapes",
    "NuImages",
    "Wildscenes",
    "RUGD",
    "Rellis3D",
    "GOOSE",
]


def browse_dataset_path():
    st.session_state.dataset_path = browse_folder()


def browse_segmentation_model_path():
    if st.session_state.get("segmentation_model_type") == "Hugging Face SegFormer":
        st.session_state.segmentation_model_path = browse_folder()
    else:
        st.session_state.segmentation_model_path = browse_file()


def browse_segmentation_config_path():
    st.session_state.segmentation_config_path = browse_file()


def browse_segmentation_ontology_path():
    st.session_state.segmentation_ontology_path = browse_file()


def render_image_segmentation_sidebar(_available_devices):
    with st.expander("Image Segmentation Dataset", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            dataset_type = st.selectbox(
                "Type",
                IMAGE_SEGMENTATION_DATASETS,
                key="segmentation_dataset_type",
            )
        with col2:
            st.selectbox("Split", ["train", "val", "test"], key="split")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_input("Dataset Folder", key="dataset_path")
        with col2:
            st.markdown(
                "<div style='margin-bottom: 1.75rem;'></div>",
                unsafe_allow_html=True,
            )
            st.button("Browse", on_click=browse_dataset_path)

        if dataset_type == "Cityscapes":
            render_cityscapes_dataset_inputs()
        else:
            st.info(f"{dataset_type} image segmentation inputs are not wired yet.")

    with st.expander("Image Segmentation Model", expanded=False):
        if st.session_state.get("segmentation_dataset_type", "Cityscapes") != "Cityscapes":
            st.info("Image segmentation model loading is wired for Cityscapes first.")
            return

        render_segmentation_model_inputs()

        if st.button(
            "Load Segmentation Model",
            type="primary",
            width="stretch",
            key="sidebar_load_segmentation_model_btn",
        ):
            load_image_segmentation_model()


def render_cityscapes_dataset_inputs():
    st.text_input(
        "Image Directory",
        value="leftImg8bit_trainvaltest/leftImg8bit",
        key="segmentation_image_dir",
    )
    st.text_input(
        "Label Directory",
        value="gtFine",
        key="segmentation_label_dir",
    )
    st.text_input(
        "Image Suffix",
        value="_leftImg8bit.png",
        key="segmentation_image_suffix",
    )
    st.text_input(
        "Label Suffix",
        value="_gtFine_labelIds.png",
        key="segmentation_label_suffix",
    )
    st.checkbox(
        "Use Train IDs",
        value=False,
        key="segmentation_use_train_id",
        help="Enable when labels use _gtFine_labelTrainIds.png.",
    )


def render_segmentation_model_inputs():
    model_type = st.selectbox(
        "Model Type",
        ["Torch Model File", "Hugging Face SegFormer"],
        key="segmentation_model_type",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_input(
            "Model Path" if model_type == "Torch Model File" else "Model Name or Folder",
            key="segmentation_model_path",
            help=(
                "Path to a TorchScript model or saved PyTorch model file."
                if model_type == "Torch Model File"
                else "Hugging Face model name or local folder downloaded with save_pretrained."
            ),
        )
    with col2:
        st.markdown(
            "<div style='margin-bottom: 1.75rem;'></div>",
            unsafe_allow_html=True,
        )
        st.button(
            "Browse",
            on_click=browse_segmentation_model_path,
            key="browse_segmentation_model_path",
        )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_input(
            "Config File",
            key="segmentation_config_path",
            help="JSON model configuration file.",
        )
    with col2:
        st.markdown(
            "<div style='margin-bottom: 1.75rem;'></div>",
            unsafe_allow_html=True,
        )
        st.button(
            "Browse",
            on_click=browse_segmentation_config_path,
            key="browse_segmentation_config_path",
        )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_input(
            "Ontology File",
            key="segmentation_ontology_path",
            help="JSON file containing the model output ontology.",
        )
    with col2:
        st.markdown(
            "<div style='margin-bottom: 1.75rem;'></div>",
            unsafe_allow_html=True,
        )
        st.button(
            "Browse",
            on_click=browse_segmentation_ontology_path,
            key="browse_segmentation_ontology_path",
        )


def load_image_segmentation_model():
    from perceptionmetrics.models.torch_segmentation import TorchImageSegmentationModel

    model_type = st.session_state.get("segmentation_model_type", "Torch Model File")
    model_path = st.session_state.get("segmentation_model_path", "")
    config_path = st.session_state.get("segmentation_config_path", "")
    ontology_path = st.session_state.get("segmentation_ontology_path", "")

    if not model_path:
        st.error("Please provide a model path or model name.")
        return
    if not config_path or not os.path.isfile(config_path):
        st.error("Please provide a valid config JSON path.")
        return
    if not ontology_path or not os.path.isfile(ontology_path):
        st.error("Please provide a valid ontology JSON path.")
        return

    with st.spinner("Loading image segmentation model..."):
        try:
            model = load_model_for_type(model_type, model_path)
            segmentation_model = TorchImageSegmentationModel(
                model=model,
                model_cfg=config_path,
                ontology_fname=ontology_path,
            )
            st.session_state.segmentation_model = segmentation_model
            st.session_state.segmentation_model_loaded = True
            st.success("Segmentation model loaded and saved for inference")
        except Exception as exc:
            st.session_state.segmentation_model = None
            st.session_state.segmentation_model_loaded = False
            st.error(f"Failed to load segmentation model: {exc}")


def load_model_for_type(model_type, model_path):
    if model_type == "Torch Model File":
        if not os.path.isfile(model_path):
            raise ValueError(
                "Torch Model File expects a .pt/.pth/.torchscript file saved with "
                "torch.save or torch.jit.save. For a Hugging Face model folder, "
                "select 'Hugging Face SegFormer'."
            )
        return model_path

    if model_type == "Hugging Face SegFormer":
        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError as exc:
            raise ImportError(
                "transformers is required for Hugging Face SegFormer models."
            ) from exc
        return SegformerForSemanticSegmentation.from_pretrained(model_path)

    raise ValueError(f"Unsupported segmentation model type: {model_type}")
