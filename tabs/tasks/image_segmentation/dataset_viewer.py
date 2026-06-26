import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from perceptionmetrics.datasets.cityscapes import CityscapesImageSegmentationDataset


def render_image_segmentation_viewer():
    dataset_type = st.session_state.get("segmentation_dataset_type", "Cityscapes")

    st.header("Dataset Viewer")

    if dataset_type == "Cityscapes":
        _render_cityscapes_viewer()
        return

    st.info(f"{dataset_type} image segmentation viewer is not wired yet.")


def _render_cityscapes_viewer():
    dataset_path = st.session_state.get("dataset_path", "")
    split = st.session_state.get("split", "val")

    if not dataset_path or not os.path.isdir(dataset_path):
        st.warning("Please select a valid Cityscapes dataset folder.")
        return

    try:
        dataset = _load_cityscapes_dataset(dataset_path, split)
    except Exception as exc:
        st.error(f"Failed to load Cityscapes dataset: {exc}")
        return

    split_df = dataset.dataset[dataset.dataset["split"] == split]
    if split_df.empty:
        st.warning(f"No Cityscapes samples found for split '{split}'.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        sample_name = st.selectbox(
            "Sample",
            split_df.index.tolist(),
            key=f"cityscapes_segmentation_sample_{split}",
        )
    with col2:
        mask_opacity = st.slider(
            "Mask Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            key="cityscapes_segmentation_mask_opacity",
        )

    row = split_df.loc[sample_name]
    image_fname = row["image"]
    label_fname = row["label"]

    try:
        image = Image.open(image_fname).convert("RGB")
        label = _read_cityscapes_label(dataset, label_fname)
    except Exception as exc:
        st.error(f"Failed to read sample '{sample_name}': {exc}")
        return

    overlay = _overlay_mask(image, label, dataset.ontology, mask_opacity)

    image_col, overlay_col = st.columns(2)
    with image_col:
        st.image(image, caption="Image", use_container_width=True)
    with overlay_col:
        st.image(overlay, caption="Ground Truth Overlay", use_container_width=True)

    with st.expander("Classes", expanded=False):
        st.dataframe(_classes_dataframe(dataset.ontology), use_container_width=True)


def _load_cityscapes_dataset(dataset_path, split):
    roots = {"train": None, "val": None, "test": None}
    roots[split] = dataset_path

    dataset_key = (
        "cityscapes_segmentation_dataset",
        os.path.abspath(dataset_path),
        split,
        st.session_state.get(
            "segmentation_image_dir", "leftImg8bit_trainvaltest/leftImg8bit"
        ),
        st.session_state.get("segmentation_label_dir", "gtFine"),
        st.session_state.get("segmentation_image_suffix", "_leftImg8bit.png"),
        st.session_state.get("segmentation_label_suffix", "_gtFine_labelIds.png"),
        st.session_state.get("segmentation_use_train_id", False),
    )

    if dataset_key not in st.session_state:
        st.session_state[dataset_key] = CityscapesImageSegmentationDataset(
            train_dataset_root=roots["train"],
            val_dataset_root=roots["val"],
            test_dataset_root=roots["test"],
            image_dir=dataset_key[3],
            label_dir=dataset_key[4],
            image_suffix=dataset_key[5],
            label_suffix=dataset_key[6],
            use_train_id=dataset_key[7],
        )

    return st.session_state[dataset_key]


def _overlay_mask(image, label, ontology, opacity):
    image_np = np.array(image)
    color_mask = _colorize_mask(label, ontology)

    resampling = getattr(Image, "Resampling", Image)
    color_mask_image = Image.fromarray(color_mask).resize(
        image.size, resampling.NEAREST
    )
    color_mask_np = np.array(color_mask_image)

    overlay = ((1.0 - opacity) * image_np + opacity * color_mask_np).astype(np.uint8)
    return Image.fromarray(overlay)


def _read_cityscapes_label(dataset, label_fname):
    if label_fname.endswith("_gtFine_color.png"):
        return _read_color_label(label_fname, dataset.ontology)

    return dataset.read_label(label_fname)


def _read_color_label(label_fname, ontology):
    label_rgb = np.array(Image.open(label_fname).convert("RGB"))
    label = np.zeros(label_rgb.shape[:2], dtype=np.uint8)

    for class_data in ontology.values():
        class_idx = int(class_data["idx"])
        rgb = np.array(class_data["rgb"], dtype=np.uint8)
        label[(label_rgb == rgb).all(axis=2)] = class_idx

    return label


def _colorize_mask(label, ontology):
    color_mask = np.zeros((*label.shape, 3), dtype=np.uint8)

    for class_data in ontology.values():
        class_idx = int(class_data["idx"])
        rgb = class_data.get("rgb", _fallback_color(class_idx))
        color_mask[label == class_idx] = rgb

    return color_mask


def _fallback_color(class_idx):
    rng = np.random.default_rng(abs(int(class_idx)))
    return tuple(int(value) for value in rng.integers(0, 255, size=3))


def _classes_dataframe(ontology):
    rows = []
    for class_name, class_data in ontology.items():
        rows.append(
            {
                "class": class_name,
                "id": class_data["idx"],
                "train_id": class_data.get("train_id"),
                "category": class_data.get("category"),
                "rgb": class_data.get("rgb"),
            }
        )
    return pd.DataFrame(rows)
