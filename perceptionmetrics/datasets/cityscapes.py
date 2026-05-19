from collections import OrderedDict
from glob import glob
import os
from typing import Optional, Tuple

import pandas as pd

from perceptionmetrics.datasets import segmentation as segmentation_dataset
import perceptionmetrics.utils.conversion as uc
from cityscapesscripts.helpers.labels import labels


def build_dataset_ontology(use_train_id:bool = False) -> dict:
    """
    Build ontology dictionary for Cityscapes dataset.

    Args:
        use_train_id: Whether to use train IDs instead of label IDs in the ontology.
    Returns:
        ontology:
            Dictionary mapping class name to:
            {"idx": label_id, "train_id": train_id, "rgb": (r, g, b)}
    """
    ontology = {}
    for label in labels:
        if label.ignoreInEval:
            continue
        ontology[label.name] = {
            "idx": label.trainId if use_train_id else label.id,
            "train_id": label.trainId,
            "cityscapes_id": label.id,
            "category": label.category,
            "category_id": label.categoryId,
            "has_instances": label.hasInstances,
            "rgb": label.color,
        }
    return ontology



def build_dataset(
    train_dataset_dir: Optional[str] = None, 
    val_dataset_dir: Optional[str] = None,
    test_dataset_dir: Optional[str] = None,
    image_dir="leftImg8bit_trainvaltest/leftImg8bit",
    label_dir : str = "gtFine_labelIds",
    image_suffix="_leftImg8bit.png",
    label_suffix="_gtFine_labelIds.png",
    use_train_id: bool = False,
) -> Tuple[dict, dict]:
    """
    Build dataset and ontology dictionaries from Cityscapes dataset structure.

    Expected structure for each provided dataset_dir:

        dataset_dir/
        ├── leftImg8bit/
        │   └── <split>/<city>/*_leftImg8bit.png
        └── gtFine/
            └── <split>/<city>/*_gtFine_labelIds.png

    Returns:
        dataset:
            OrderedDict mapping sample_name to:
            (image_path, label_path, city, split)

        ontology:
            Dictionary mapping class name to:
            {"idx": label_id, "train_id": train_id, "rgb": (r, g, b)}
    """
    
    # Define dataset directories and ensure they are absolute paths
    dataset_dirs = {
        "train": train_dataset_dir,
        "val": val_dataset_dir,
        "test": test_dataset_dir,
    }
    dataset_dirs = {
        split: os.path.abspath(d) for split, d in dataset_dirs.items() if d is not None
    }
    if not dataset_dirs:
        raise ValueError("At least one dataset directory must be provided")
    
    ontology = build_dataset_ontology(use_train_id=use_train_id)
    dataset = OrderedDict()
    
    for split, dataset_dir in dataset_dirs.items():

        image_pattern = os.path.join(
            dataset_dir,
            image_dir,
            split,
            "*",
            f"*{image_suffix}",
        )

        image_fnames = sorted(glob(image_pattern))

        if len(image_fnames) == 0:
            print(f"No images found for split={split}")
            print(f"Searched pattern: {image_pattern}")
            continue

        for image_fname in image_fnames:
            image_fname = os.path.abspath(image_fname)

            city = os.path.basename(os.path.dirname(image_fname))
            image_basename = os.path.basename(image_fname)

            sample_name = image_basename.replace(image_suffix, "")

            label_basename = f"{sample_name}{label_suffix}"

            label_fname = os.path.join(
                dataset_dir,
                label_dir,
                split,
                city,
                label_basename,
            )
            label_fname = os.path.abspath(label_fname)

            if not os.path.exists(label_fname):
                print(f"Missing label for image: {image_fname}")
                print(f"Expected label path: {label_fname}")
                continue

            dataset[sample_name] = (
                image_fname,
                label_fname,
                city,
                split,
            )

    return dataset, ontology
        
        
cityscapes_dir = "/home/tejass/Downloads/TUDELFT_ROBOTICS/GSOC/gsoc2026-Tejas_Stanley/datasets/cityscapes"

dataset, ontology = build_dataset(
    train_dataset_dir=cityscapes_dir,
    val_dataset_dir=cityscapes_dir,
    image_dir="leftImg8bit_trainvaltest/leftImg8bit",
    label_dir="gtFine",
    image_suffix="_leftImg8bit.png",
    label_suffix="_gtFine_labelIds.png",
    use_train_id=False,
)

#DEBUG Build_dataset and build_ontology functions
# for i, (sample_name, sample_data) in enumerate(dataset.items()):
#     if i == 5:
#         break

#     print(sample_name)
#     print(sample_data)
#     print()



