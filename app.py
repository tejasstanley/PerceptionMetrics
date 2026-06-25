import streamlit as st

from perceptionmetrics.utils.torch import get_device_info
from tabs.dataset_viewer import dataset_viewer_tab
from tabs.evaluator import evaluator_tab
from tabs.inference import inference_tab
from tabs.tasks.image_detection import render_image_detection_sidebar


st.set_page_config(page_title="PerceptionMetrics", layout="wide")

PAGES = {
    "Dataset Viewer": dataset_viewer_tab,
    "Inference": inference_tab,
    "Evaluator": evaluator_tab,
}

best_device, available_devices = get_device_info()

# Shared state
st.session_state.setdefault("task", "Image Detection")
st.session_state.setdefault("dataset_path", "")
st.session_state.setdefault("split", "test")
st.session_state.setdefault("device", best_device)

# Image detection state
st.session_state.setdefault("dataset_type", "YOLO")
st.session_state.setdefault("config_option", "Manual Configuration")
st.session_state.setdefault("confidence_threshold", 0.5)
st.session_state.setdefault("nms_threshold", 0.5)
st.session_state.setdefault("max_detections", 100)
st.session_state.setdefault("batch_size", 1)
st.session_state.setdefault("evaluation_step", 5)
st.session_state.setdefault("detection_model", None)
st.session_state.setdefault("detection_model_loaded", False)

with st.sidebar:
    task = st.selectbox(
        "Task",
        ["Image Detection"],
        key="task",
        help="Only image detection is currently wired in the GUI.",
    )

    if task == "Image Detection":
        render_image_detection_sidebar(available_devices)
    else:
        st.error(f"Unsupported task: {task}")


tab1, tab2, tab3 = st.tabs(["Dataset Viewer", "Inference", "Evaluator"])

with tab1:
    dataset_viewer_tab()
with tab2:
    inference_tab()
with tab3:
    evaluator_tab()
