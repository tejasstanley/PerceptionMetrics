from collections import OrderedDict
from glob import glob
import logging
import os
from typing import List, Optional, Tuple, Union

import pandas as pd
from perceptionmetrics.datasets import segmentation as segmentation_dataset
import perceptionmetrics.utils.io as uio


def _find_sequence_root(dataset_dir: str, subdir: str) -> Optional[str]:
    """Find the SemanticKITTI sequence root containing the requested subdirectory.
    :param dataset_dir: Directory where SemanticKITTI data has been extracted to
    :type dataset_dir: str
    :param subdir: Subdirectory to look for (e.g., "velodyne" or "labels")
    :type subdir: str
    :return: Path to the sequence root containing the requested subdirectory, or None if not found
    :rtype: Optional[str]
    """
    candidates = [
        os.path.join(dataset_dir, "dataset", "sequences"),
        os.path.join(dataset_dir, "sequences"),
        os.path.join(dataset_dir, f"data_odometry_{subdir}", "dataset", "sequences"),
    ]

    for candidate in candidates:
        if glob(os.path.join(candidate, "*", subdir)):
            return candidate

    matches = glob(
        os.path.join(dataset_dir, "**", "dataset", "sequences"), recursive=True
    )
    matches.extend(glob(os.path.join(dataset_dir, "**", "sequences"), recursive=True))
    for match in matches:
        if glob(os.path.join(match, "*", subdir)):
            return match

    return None


def _build_ontology(config: dict) -> dict:
    """Build a compact ontology from SemanticKITTI's learning-map config.
    :param config: SemanticKITTI YAML config containing ontology and splits
    :type config: dict
    :return: Compact ontology with class names as keys and dicts with 'idx' and 'rgb' as values
    :rtype: dict
    """
    labels = config["labels"]
    color_map = config["color_map"]
    learning_map_inv = config["learning_map_inv"]
    content = config.get("content", {})

    ontology = OrderedDict()
    for idx in sorted(learning_map_inv):
        raw_idx = learning_map_inv[idx]
        class_name = labels[raw_idx]
        bgr = color_map[raw_idx]
        class_data = {
            "idx": idx,
            "rgb": tuple(reversed(bgr)),
        }
        if raw_idx in content:
            class_data["content"] = content[raw_idx]
        ontology[class_name] = class_data

    return ontology


def build_dataset(
    dataset_dir: str,
    config_fname: str,
    split: Optional[Union[str, List[str]]] = None,
) -> Tuple[dict, dict]:
    """Build dataset and ontology dictionaries from SemanticKITTI dataset structure

    :param dataset_dir: Directory where SemanticKITTI data has been extracted to
    :type dataset_dir: str
    :param config_fname: SemanticKITTI YAML config containing ontology and splits
    :type config_fname: str
    :param split: Split or splits to load. Loads all splits if None, defaults to None
    :type split: Optional[Union[str, List[str]]], optional
    :return: Dataset and ontology
    :rtype: Tuple[dict, dict]
    """
    dataset_dir = os.path.abspath(dataset_dir)
    print(f"Building dataset from SemanticKITTI directory: {dataset_dir}")
    assert os.path.isdir(dataset_dir), "Dataset directory not found"
    assert os.path.isfile(config_fname), "SemanticKITTI config file not found"

    config = uio.read_yaml(config_fname)
    ontology = _build_ontology(config)
    requested_splits = [split] if isinstance(split, str) else split

    points_sequence_root = _find_sequence_root(dataset_dir, "velodyne")
    label_sequence_root = _find_sequence_root(dataset_dir, "labels")
    assert (
        points_sequence_root is not None
    ), "SemanticKITTI velodyne directory not found"

    dataset = OrderedDict()
    skipped_samples = []
    missing_label_count = 0
    total_samples = 0
    requested_split_names = (
        ["valid" if s == "val" else s for s in requested_splits]
        if requested_splits is not None
        else None
    )

    for config_split, sequences in config["split"].items():
        split_name = "val" if config_split == "valid" else config_split
        if (
            requested_split_names is not None
            and config_split not in requested_split_names
        ):
            continue

        for sequence in sequences:
            scene = f"{sequence:02d}"
            points_dir = os.path.join(points_sequence_root, scene, "velodyne")
            label_dir = (
                os.path.join(label_sequence_root, scene, "labels")
                if label_sequence_root is not None
                else None
            )

            if not os.path.isdir(points_dir):
                logging.warning(
                    f"Missing velodyne directory for sequence {scene}. Skipped!"
                )
                continue

            points_fnames = sorted(glob(os.path.join(points_dir, "*.bin")))
            total_samples += len(points_fnames)
            for points_fname in points_fnames:
                sample_base_name, _ = os.path.splitext(os.path.basename(points_fname))
                sample_name = f"{scene}-{sample_base_name}"
                label_fname = (
                    os.path.join(label_dir, f"{sample_base_name}.label")
                    if label_dir is not None
                    else None
                )

                if label_fname is None or not os.path.isfile(label_fname):
                    if split_name == "test":
                        label_fname = None
                    else:
                        missing_label_count += 1
                        logging.warning(f"Missing label for {sample_name}. Skipped!")
                        skipped_samples.append(sample_name)
                        continue

                dataset[sample_name] = (points_fname, label_fname, scene, split_name)

    print(f"Samples retrieved: {len(dataset)} / {total_samples}")
    if missing_label_count:
        print(f"Samples without labels: {missing_label_count}")
    if skipped_samples:
        print("Skipped samples:")
        for sample_name in skipped_samples:
            print(f"\n\t{sample_name}")

    return dataset, ontology


class SemanticKITTILiDARSegmentationDataset(
    segmentation_dataset.LiDARSegmentationDataset
):
    """Specific class for SemanticKITTI-styled LiDAR segmentation datasets.

    :param dataset_dir: Directory where SemanticKITTI data has been extracted to
    :type dataset_dir: str
    :param config_fname: SemanticKITTI YAML config containing ontology and splits
    :type config_fname: str
    :param split: Split or splits to load. Accepts "train", "val"/"valid", and "test".
        Loads all splits if None, defaults to None
    :type split: Optional[Union[str, List[str]]], optional
    """

    def __init__(
        self,
        dataset_dir: str,
        config_fname: str,
        split: Optional[Union[str, List[str]]] = None,
    ):
        dataset, ontology = build_dataset(dataset_dir, config_fname, split=split)

        cols = ["points", "label", "scene", "split"]
        dataset = pd.DataFrame.from_dict(dataset, orient="index", columns=cols)
        dataset.attrs = {"ontology": ontology}

        super().__init__(dataset, dataset_dir, ontology)


if __name__ == "__main__":
    dataset = SemanticKITTILiDARSegmentationDataset(
        "local/data/SemanticKITTI",
        "local/data/SemanticKITTI/semantic-kitti.yaml",
        split="val",
    )

    print(dataset.dataset["split"].value_counts().to_dict())
    print(dataset.dataset.head())
