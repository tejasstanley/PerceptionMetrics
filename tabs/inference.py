import streamlit as st


def inference_tab():
    task = st.session_state.get("task", "Image Detection")

    if task == "Image Detection":
        from tabs.tasks.image_detection import render_image_detection_inference

        render_image_detection_inference()
        return

    st.error(f"Unsupported task for inference: {task}")
