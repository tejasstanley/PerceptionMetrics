__all__ = [
    "render_image_detection_sidebar",
    "render_image_detection_viewer",
    "render_image_detection_inference",
    "render_image_detection_evaluator",
]


def __getattr__(name):
    if name == "render_image_detection_sidebar":
        from tabs.tasks.image_detection.sidebar import render_image_detection_sidebar

        return render_image_detection_sidebar
    if name == "render_image_detection_viewer":
        from tabs.tasks.image_detection.dataset_viewer import render_image_detection_viewer

        return render_image_detection_viewer
    if name == "render_image_detection_inference":
        from tabs.tasks.image_detection.inference import render_image_detection_inference

        return render_image_detection_inference
    if name == "render_image_detection_evaluator":
        from tabs.tasks.image_detection.evaluator import render_image_detection_evaluator

        return render_image_detection_evaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
