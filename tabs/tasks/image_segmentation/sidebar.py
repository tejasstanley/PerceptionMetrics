import streamlit as st

from perceptionmetrics.utils.gui import browse_folder


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
            _render_cityscapes_dataset_inputs()
        else:
            st.info(f"{dataset_type} image segmentation inputs are not wired yet.")

    with st.expander("Image Segmentation Model", expanded=False):
        st.info("Image segmentation model loading is a placeholder for now.")


def _render_cityscapes_dataset_inputs():
    st.text_input(
        "Image Directory",
        value="leftImg8bit_trainvaltest/leftImg8bit",
        key="segmentation_image_dir",
    )
    st.text_input("Label Directory", value="gtFine", key="segmentation_label_dir")
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
