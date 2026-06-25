import streamlit as st


def evaluator_tab():
    task = st.session_state.get("task", "Image Detection")

    if task == "Image Detection":
        from tabs.tasks.image_detection import render_image_detection_evaluator

        render_image_detection_evaluator()
        return

    st.error(f"Unsupported task for evaluator: {task}")
