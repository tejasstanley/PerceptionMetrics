import os
import tempfile

import streamlit as st

from perceptionmetrics.datasets.coco import CocoDataset, find_img_dir_and_ann_file
from perceptionmetrics.utils.gui import browse_folder
from tabs.tasks.image_detection.sidebar import _uploaded_json_to_tempfile


def browse_predictions_outdir():
    folder = browse_folder()
    if folder:
        st.session_state.predictions_outdir = folder


def render_image_detection_evaluator():
    st.header("Evaluator")
    st.markdown("Evaluate your model on the loaded dataset using PerceptionMetrics.")

    dataset_available, model_available, dataset, model = _get_eval_dataset_and_model()

    st.markdown("### Evaluation Configuration")

    save_predictions = st.checkbox(
        "Save Predictions",
        value=False,
        help="Save individual predictions and metrics per sample",
    )

    save_visualizations = st.checkbox(
        "Save Visualizations",
        value=False,
        help="Save visualized qualitative results (Image + GT + Preds)",
    )

    predictions_outdir_input = None
    if save_predictions or save_visualizations:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_input("Predictions Output Directory", key="predictions_outdir")
        with col2:
            st.markdown(
                "<div style='margin-bottom: 1.75rem;'></div>",
                unsafe_allow_html=True,
            )
            st.button(
                "Browse", on_click=browse_predictions_outdir, key="browse_preds_outdir"
            )
        predictions_outdir_input = st.session_state.get("predictions_outdir")

    ontology_translation = st.file_uploader(
        "Ontology Translation (Optional)",
        type=["json"],
        help="JSON file for translating between dataset and model ontologies",
    )

    output_dir_required = save_predictions or save_visualizations
    output_dir_missing = output_dir_required and not (
        predictions_outdir_input and predictions_outdir_input.strip()
    )

    if output_dir_missing:
        st.warning(
            "Please provide a Predictions Output Directory to enable evaluation "
            "when 'Save Predictions' or 'Save Visualizations' is turned on."
        )

    if st.button(
        "Run Evaluation",
        type="primary",
        disabled=not (dataset_available and model_available) or output_dir_missing,
    ):
        _run_detection_evaluation(
            dataset,
            model,
            ontology_translation,
            save_predictions,
            save_visualizations,
            predictions_outdir_input,
        )

    if "evaluation_results" in st.session_state:
        display_evaluation_results(st.session_state["evaluation_results"])

def _get_eval_dataset_and_model():
    dataset_available = False
    model_available = False
    dataset = None
    model = None

    dataset_path = st.session_state.get("dataset_path", "")
    dataset_type = st.session_state.get("dataset_type", "Coco")
    split = st.session_state.get("split", "val")
    dataset_key = f"{dataset_path}_{split}"

    if dataset_key in st.session_state:
        dataset = st.session_state[dataset_key]
        dataset_available = True
        st.success(
            f"Dataset loaded: {dataset_path} ({split} split) - {len(dataset.dataset)} samples"
        )
    elif dataset_path and os.path.isdir(dataset_path):
        try:
            if dataset_type.lower() == "coco":
                img_dir, ann_file = find_img_dir_and_ann_file(
                    dataset_path=dataset_path, split=split
                )

                if os.path.isdir(img_dir) and os.path.isfile(ann_file):
                    st.session_state[dataset_key] = CocoDataset(
                        annotation_file=ann_file, image_dir=img_dir, split=split
                    )
                    st.session_state[dataset_key].make_fname_global()
                    dataset = st.session_state[dataset_key]
                    dataset_available = True
                    st.success(
                        f"Dataset loaded: {dataset_path} ({split} split) - {len(dataset.dataset)} samples"
                    )
                else:
                    st.warning(
                        "Dataset files not found. Please check the dataset path and split in the sidebar."
                    )
            else:
                st.warning("Only COCO datasets are currently supported for evaluation.")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
    else:
        st.warning("No dataset path provided. Please set the dataset path in the sidebar.")

    if (
        "detection_model" in st.session_state
        and st.session_state.detection_model is not None
    ):
        model = st.session_state.detection_model
        model_available = True
        st.success("Model loaded and ready for evaluation")
    else:
        st.warning("No model loaded. Please load a model using the sidebar.")

    return dataset_available, model_available, dataset, model

def _run_detection_evaluation(
    dataset,
    model,
    ontology_translation,
    save_predictions,
    save_visualizations,
    predictions_outdir_input,
):
    split = st.session_state.get("split", "val")
    with st.spinner("Running evaluation..."):
        try:
            if len(dataset.dataset) == 0:
                st.error("Dataset has no samples. Please check the dataset configuration.")
                return

            if not hasattr(model, "model_cfg") or model.model_cfg is None:
                st.error("Model configuration is missing. Please reload the model.")
                return

            ontology_translation_path = None
            if ontology_translation is not None:
                ontology_translation_path = _uploaded_json_to_tempfile(ontology_translation)

            predictions_outdir = None
            if save_predictions or save_visualizations:
                if predictions_outdir_input and predictions_outdir_input.strip():
                    predictions_outdir = predictions_outdir_input.strip()
                    os.makedirs(predictions_outdir, exist_ok=True)
                else:
                    predictions_outdir = tempfile.mkdtemp(prefix="eval_predictions_")

            progress_bar = st.progress(0)
            status_text = st.empty()
            intermediate_metrics_placeholder = st.empty()
            intermediate_table_placeholder = st.empty()

            def progress_callback(processed, total):
                try:
                    progress = processed / total if total > 0 else 0
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Processing: {processed}/{total} images ({progress:.1%})"
                    )
                except Exception as e:
                    st.error(f"Progress callback error: {e}")

            def metrics_callback(metrics_df, processed, total):
                _render_intermediate_detection_metrics(
                    metrics_df,
                    processed,
                    intermediate_metrics_placeholder,
                    intermediate_table_placeholder,
                )

            try:
                results = model.eval(
                    dataset=dataset,
                    split=split,
                    ontology_translation=ontology_translation_path,
                    predictions_outdir=predictions_outdir,
                    results_per_sample=save_predictions,
                    save_visualizations=save_visualizations,
                    progress_callback=progress_callback,
                    metrics_callback=metrics_callback,
                )
            except Exception as e:
                st.error(f"Error in model.eval(): {e}")
                return

            progress_bar.empty()
            status_text.empty()
            intermediate_metrics_placeholder.empty()
            intermediate_table_placeholder.empty()

            st.session_state["evaluation_results"] = results
            st.session_state["evaluation_config"] = {
                "split": split,
                "predictions_saved": save_predictions,
                "visualizations_saved": save_visualizations,
            }

            st.success("Evaluation completed successfully!")
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            import traceback

            st.code(traceback.format_exc())

def _render_intermediate_detection_metrics(
    metrics_df,
    processed,
    intermediate_metrics_placeholder,
    intermediate_table_placeholder,
):
    try:
        if "mean" in metrics_df.columns:
            mean_metrics = metrics_df["mean"]

            with intermediate_metrics_placeholder.container():
                st.markdown(f"#### Intermediate Results (after {processed} images)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("mAP", f"{mean_metrics.get('AP', 0):.3f}")
                with col2:
                    st.metric(
                        "Mean Precision",
                        f"{mean_metrics.get('Precision', 0):.3f}",
                    )
                with col3:
                    st.metric("Mean Recall", f"{mean_metrics.get('Recall', 0):.3f}")

        per_class_results = (
            metrics_df.drop(columns=["mean"])
            if "mean" in metrics_df.columns
            else metrics_df
        )
        per_class_results = per_class_results.drop(
            ["AUC-PR", "mAP@[0.5:0.95]"], errors="ignore"
        )

        display_df = per_class_results.copy()
        numeric_columns = display_df.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(3)

        with intermediate_table_placeholder.container():
            st.markdown("#### Per-Class Metrics (Intermediate)")
            st.dataframe(display_df, width="stretch")
    except Exception as e:
        st.error(f"Metrics callback error: {e}")

def display_evaluation_results(results):
    if results is None:
        st.warning("No evaluation results to display.")
        return

    if isinstance(results, dict):
        metrics_df = results.get("metrics_df")
        metrics_factory = results.get("metrics_factory")
    else:
        metrics_df = results
        metrics_factory = None

    if metrics_df is None or metrics_df.empty:
        st.warning("No evaluation results to display.")
        return

    st.markdown("#### Summary Metrics")

    if "mean" in metrics_df.columns:
        mean_metrics = metrics_df["mean"]
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("mAP", f"{mean_metrics.get('AP', 0):.3f}")
        with col2:
            st.metric("Mean Precision", f"{mean_metrics.get('Precision', 0):.3f}")
        with col3:
            st.metric("Mean Recall", f"{mean_metrics.get('Recall', 0):.3f}")
        with col4:
            coco_map = mean_metrics.get("mAP@[0.5:0.95]", 0)
            st.metric("mAP@[0.5:0.95]", f"{coco_map:.3f}")
        with col5:
            auc_pr = mean_metrics.get("AUC-PR", 0)
            st.metric("AUC-PR", f"{auc_pr:.3f}")

    st.markdown("#### Per-Class Metrics")
    per_class_results = (
        metrics_df.drop(columns=["mean"])
        if "mean" in metrics_df.columns
        else metrics_df
    )
    per_class_results = per_class_results.drop(
        ["AUC-PR", "mAP@[0.5:0.95]"], errors="ignore"
    )

    display_df = per_class_results.copy()
    numeric_columns = display_df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(3)

    st.dataframe(display_df, width="stretch")

    if metrics_factory is not None:
        _render_precision_recall_curve(metrics_factory)

    st.markdown("#### Download Results")
    st.download_button(
        label="Download per class metrics",
        data=metrics_df.to_csv(index=True),
        file_name="evaluation_results.csv",
        mime="text/csv",
    )
    _render_pr_points_download(metrics_factory)
    _render_detailed_detection_statistics(metrics_df, metrics_factory)

def _render_precision_recall_curve(metrics_factory):
    st.markdown("#### Precision-Recall Curve")
    try:
        import plotly.graph_objects as go

        curve_data = metrics_factory.get_overall_precision_recall_curve()
        auc_pr = metrics_factory.compute_auc_pr()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=curve_data["recall"],
                y=curve_data["precision"],
                mode="lines",
                name="Precision-Recall Curve",
                line=dict(color="blue", width=2),
                fill="tonexty",
                fillcolor="rgba(0, 0, 255, 0.1)",
            )
        )
        fig.add_annotation(
            x=0.6,
            y=0.2,
            text=f"AUC-PR: {auc_pr:.3f}",
            showarrow=False,
            font=dict(size=12),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
        )
        fig.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            showlegend=True,
            height=500,
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.error(f"Error plotting precision-recall curve: {e}")
        st.info("Precision-recall curve data not available.")

def _render_pr_points_download(metrics_factory):
    try:
        curve_data = (
            metrics_factory.get_overall_precision_recall_curve()
            if metrics_factory is not None
            else None
        )
        if curve_data is not None:
            import pandas as pd

            pr_points_df = pd.DataFrame(
                {"recall": curve_data["recall"], "precision": curve_data["precision"]}
            )
            st.download_button(
                label="Download precision-recall points",
                data=pr_points_df.to_csv(index=False),
                file_name="precision_recall_points.csv",
                mime="text/csv",
            )
        else:
            st.write("No precision-recall data available.")
    except Exception as e:
        st.write(f"Error preparing precision-recall points: {e}")

def _render_detailed_detection_statistics(metrics_df, metrics_factory):
    with st.expander("Detailed Statistics"):
        st.markdown("**Results Shape:**")
        st.write(f"Rows: {metrics_df.shape[0]}, Columns: {metrics_df.shape[1]}")

        st.markdown("**Available Metrics:**")
        st.write(list(metrics_df.columns))

        st.markdown("**Class Names:**")
        st.write(
            list(metrics_df.index) if len(metrics_df.index) > 0 else "No classes found"
        )

        st.markdown("**DataFrame Info:**")
        st.write("Index:", metrics_df.index.tolist())
        st.write("Columns:", metrics_df.columns.tolist())

        st.markdown("**Sample Data:**")
        st.dataframe(metrics_df.head(), width="stretch")

        if "evaluation_config" in st.session_state:
            st.markdown("**Evaluation Configuration:**")
            config = st.session_state["evaluation_config"]
            for key, value in config.items():
                st.write(f"- {key}: {value}")

        if metrics_factory is not None:
            st.markdown("**Precision-Recall Curve Data:**")
            try:
                curve_data = metrics_factory.get_overall_precision_recall_curve()
                st.write(f"Number of points: {len(curve_data['precision'])}")
                st.write(
                    f"Precision range: {min(curve_data['precision']):.3f} - {max(curve_data['precision']):.3f}"
                )
                st.write(
                    f"Recall range: {min(curve_data['recall']):.3f} - {max(curve_data['recall']):.3f}"
                )
                st.write(f"AUC-PR: {metrics_factory.compute_auc_pr():.3f}")
            except Exception as e:
                st.write(f"Error accessing curve data: {e}")
