import os
import tempfile

import streamlit as st
from PIL import Image

from perceptionmetrics.datasets.coco import CocoDataset, find_img_dir_and_ann_file


def render_image_detection_viewer():
    import numpy as np
    from perceptionmetrics.datasets.yolo import YOLODataset
    from streamlit_image_select import image_select
    from supervision.detection.annotate import BoxAnnotator
    from supervision.detection.core import Detections
    from supervision.draw.color import ColorPalette

    dataset_path = st.session_state.get("dataset_path", "")
    dataset_type = st.session_state.get("dataset_type", "COCO").lower()
    split = st.session_state.get("split", "val")

    st.header("Dataset Viewer")

    if not dataset_path or not os.path.isdir(dataset_path):
        st.warning("Please select a valid dataset folder.")
        return

    if dataset_type == "coco":
        try:
            img_dir, ann_file = find_img_dir_and_ann_file(
                dataset_path=dataset_path, split=split
            )
        except FileNotFoundError:
            st.warning("Dataset files not found. Check path and split.")
            return
    elif dataset_type == "yolo":
        dataset_config_file = st.session_state.get("dataset_config_file", None)
        img_dir = os.path.join(dataset_path, f"images/{split}")
        if not os.path.isdir(img_dir):
            st.warning("Image directory not found.")
            return
        if dataset_config_file is None:
            st.warning("Dataset configuration file not found. Please upload it.")
            return
    else:
        st.error("Unsupported dataset type.")
        return

    _render_detection_viewer_style()
    dataset = _load_detection_dataset(dataset_path, dataset_type, split, img_dir, ann_file if dataset_type == "coco" else None)
    if dataset is None:
        return

    image_files = [
        f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not image_files:
        st.warning("No images found.")
        return

    image_paths, sample_images, current_page = _render_image_grid_navigation(
        dataset_path, split, image_files, img_dir
    )
    captions = [_short_caption(name) for name in sample_images]

    img_select_key = f"img_select_all_{dataset_path}_{split}_{current_page}"
    img_select_index = st.session_state.get(img_select_key)
    if img_select_index is None or not isinstance(img_select_index, int):
        img_select_index = 0
    selected_img_path = (
        image_select(
            label="",
            images=image_paths,
            captions=captions,
            use_container_width=False,
            key=img_select_key,
            index=img_select_index,
        )
        if image_paths
        else None
    )

    if selected_img_path:
        selected_img_name = os.path.basename(selected_img_path)
        try:
            img = Image.open(selected_img_path).convert("RGB")
            img_np = np.array(img)

            if dataset_type == "yolo":
                ann_row = dataset.dataset[
                    dataset.dataset["image"].str.endswith(selected_img_name)
                ]
            else:
                ann_row = dataset.dataset[dataset.dataset["image"] == selected_img_name]

            if not ann_row.empty:
                annotation_id = ann_row.iloc[0]["annotation"]
                if dataset_type == "yolo":
                    annotation_id = os.path.join(dataset.dataset_dir, annotation_id)

                boxes, category_indices = dataset.read_annotation(annotation_id)
                class_names = _class_names_from_ontology(dataset, category_indices)

                palette = ColorPalette.default()
                detections = Detections(
                    xyxy=np.array(boxes), class_id=np.array(category_indices)
                )
                annotator = BoxAnnotator(
                    color=palette, text_scale=0.7, text_thickness=1, text_padding=2
                )
                annotated_img = annotator.annotate(
                    scene=img_np, detections=detections, labels=class_names
                )

                annotated_pil = Image.fromarray(annotated_img)
                resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                annotated_pil.thumbnail((500, 500), resample)
                st.image(annotated_pil, width="content")
            else:
                st.warning("No annotation found for this image.")
        except Exception as e:
            st.error(f"Error displaying image: {e}")
    else:
        st.info("Select an image to view with annotations.")

def _render_detection_viewer_style():
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 2, 1.5])
    with nav_col1:
        pass
    with nav_col2:
        pass
    with nav_col3:
        pass
    with nav_col4:
        st.markdown(
            """
            <style>
            div[data-testid="stButton"] button#search_icon_btn {
                padding: 0.15rem 0.5rem;
                font-size: 0.85rem;
                min-height: 1.5rem;
                height: 1.5rem;
                line-height: 1.1;
            }
            div[data-testid="stButton"] {
                margin-top: -0.85rem !important;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-bottom: 0;'></div>", unsafe_allow_html=True)

def _load_detection_dataset(dataset_path, dataset_type, split, img_dir, ann_file):
    from perceptionmetrics.datasets.yolo import YOLODataset

    dataset_key = f"{dataset_path}_{split}"
    if dataset_key not in st.session_state:
        try:
            if dataset_type == "coco":
                st.session_state[dataset_key] = CocoDataset(
                    annotation_file=ann_file,
                    image_dir=img_dir,
                    split=split,
                )
            elif dataset_type == "yolo":
                dataset_config_file = st.session_state.get("dataset_config_file", None)
                if dataset_config_file is None:
                    st.warning("Dataset configuration file not found. Please upload it.")
                    return None
                with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as tmp:
                    tmp.write(dataset_config_file.read())
                    tmp_path = tmp.name

                yolo_dataset = YOLODataset(tmp_path, dataset_path)
                st.session_state["full_dataset_df"] = yolo_dataset.dataset
                yolo_dataset.dataset = yolo_dataset.dataset[
                    yolo_dataset.dataset["split"] == split
                ].reset_index(drop=True)
                st.session_state[dataset_key] = yolo_dataset
                os.unlink(tmp_path)
            else:
                st.error("Unsupported dataset type.")
                return None
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            return None
    else:
        try:
            cached_ds = st.session_state[dataset_key]
            cached_split = getattr(cached_ds, "split", None)
            if cached_split != split:
                if dataset_type == "coco":
                    st.session_state[dataset_key] = CocoDataset(
                        annotation_file=ann_file,
                        image_dir=img_dir,
                        split=split,
                    )
                elif dataset_type == "yolo":
                    yolo_dataset = st.session_state[dataset_key]
                    yolo_dataset.dataset = st.session_state["full_dataset_df"][
                        st.session_state["full_dataset_df"]["split"] == split
                    ].reset_index(drop=True)
                    st.session_state[dataset_key] = yolo_dataset
        except Exception:
            pass
    return st.session_state[dataset_key]

def _render_image_grid_navigation(dataset_path, split, image_files, img_dir):
    images_per_page = 12
    total_pages = (len(image_files) + images_per_page - 1) // images_per_page
    page_key = f"image_page_{dataset_path}_{split}"

    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    current_page = max(0, min(st.session_state[page_key], total_pages - 1))
    st.session_state[page_key] = current_page

    start_idx = current_page * images_per_page
    sample_images = image_files[start_idx : start_idx + images_per_page]
    image_paths = [os.path.join(img_dir, img_name) for img_name in sample_images]

    col1, col2, col3, col4 = st.columns([0.5, 9.5, 0.5, 0.5])
    with col1:
        if st.button("⟨", key="prev_page_btn", disabled=(current_page == 0)):
            st.session_state[page_key] = current_page - 1
            st.rerun()
    with col2:
        st.markdown(
            f"<div style='text-align:center;font-weight:bold;'>Page {current_page + 1} of {total_pages}</div>",
            unsafe_allow_html=True,
        )
    with col3:
        if st.button(
            "⟩", key="next_page_btn", disabled=(current_page >= total_pages - 1)
        ):
            st.session_state[page_key] = current_page + 1
            st.rerun()
    with col4:
        if st.button(
            "🔍",
            key="search_icon_btn",
            help="Search for an image by name",
            disabled=not (dataset_path and os.path.isdir(dataset_path)),
        ):
            st.session_state["show_search_dropdown"] = True

    if st.session_state.get("show_search_dropdown", False):
        _render_image_search_dropdown(image_files, images_per_page, page_key, dataset_path, split)

    return image_paths, sample_images, current_page

def _render_image_search_dropdown(image_files, images_per_page, page_key, dataset_path, split):
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        selected_img = st.selectbox("Search image:", options=image_files, key="search_image")
    with col2:
        st.markdown("<div style='margin-bottom: 2.4rem;'></div>", unsafe_allow_html=True)
        if st.button("Go to image", key="go_to_image_btn"):
            new_page = image_files.index(selected_img) // images_per_page
            st.session_state[page_key] = new_page
            st.session_state[f"img_select_all_{dataset_path}_{split}_{new_page}"] = (
                image_files.index(selected_img) % images_per_page
            )
            st.session_state["show_search_dropdown"] = False
            st.rerun()
    with col3:
        st.markdown("<div style='margin-bottom: 2.4rem;'></div>", unsafe_allow_html=True)
        if st.button("Cancel", key="cancel_search_btn"):
            st.session_state["show_search_dropdown"] = False
            st.rerun()

def _short_caption(name: str) -> str:
    caption_len_limit = 17
    if len(name) > caption_len_limit:
        return name[:caption_len_limit] + "..." + name[-3:]
    return name

def _class_names_from_ontology(dataset, category_indices):
    ontology = getattr(dataset, "ontology", None)
    if ontology is None and hasattr(dataset.dataset, "attrs"):
        ontology = dataset.dataset.attrs.get("ontology", None)

    if ontology:
        catid_to_name = {v["idx"]: k for k, v in ontology.items()}
        return [catid_to_name.get(cat_id, str(cat_id)) for cat_id in category_indices]
    return [str(cat_id) for cat_id in category_indices]

