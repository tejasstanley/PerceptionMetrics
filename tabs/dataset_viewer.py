import streamlit as st


def dataset_viewer_tab():
    task = st.session_state.get("task", "Image Detection")

    if task == "Image Detection":
        from tabs.tasks.image_detection import render_image_detection_viewer

        render_image_detection_viewer()
        return

    st.error(f"Unsupported task for dataset viewer: {task}")
