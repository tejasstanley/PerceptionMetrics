---
layout: home
title: Datasets
permalink: /datasets/

sidebar:
  nav: "main"
---

This page summarizes the dataset adapters available in PerceptionMetrics and the inputs each adapter expects. Dataset adapters normalize different dataset layouts into the common `PerceptionDataset` abstractions used by model evaluation, prediction evaluation, the GUI, and the Python API.

## Support Matrix

| Dataset adapter | Task | Modality | CLI format | Status |
| --- | --- | --- | --- | --- |
| GAIA | Segmentation | Image, LiDAR | `gaia` | CLI and library |
| Generic | Segmentation | Image, LiDAR | `generic` | CLI and library |
| GOOSE | Segmentation | Image, LiDAR | `goose` | CLI and library |
| RELLIS-3D | Segmentation | Image, LiDAR | `rellis3d` | CLI and library |
| RUGD | Segmentation | Image | `rugd` | CLI and library |
| WildScenes | Segmentation | Image, LiDAR | `wildscenes` | Library adapter |
| Cityscapes | Segmentation | Image | `cityscapes` | Library adapter |
| COCO | Object detection | Image | `coco` | CLI and library |
| YOLO | Object detection | Image | `yolo` | Library adapter |
| nuImages | Segmentation, object detection | Image | `nuimages` | Library adapter |

The CLI currently constructs datasets through `perceptionmetrics.cli.get_dataset`. If a dataset appears in the library registry but is not handled by that CLI helper, use it from Python until CLI wiring is added.

## Common Concepts

All segmentation datasets expose samples with a data file, label file, and split. Image segmentation datasets use image masks as labels; LiDAR segmentation datasets use point labels. Object detection datasets expose image files and annotations that can be converted into bounding boxes and class indices.

Most segmentation adapters need an ontology. The common ontology shape is:

```json
{
  "road": {
    "idx": 0,
    "rgb": [128, 64, 128]
  },
  "vegetation": {
    "idx": 1,
    "rgb": [107, 142, 35]
  }
}
```

Some adapters build the ontology from the official dataset metadata. For others, you need to provide an ontology file. The ontology file can be in JSON or YAML format, and the exact expected shape depends on the adapter. 



## GAIA

GAIA is a custom PerceptionMetrics format backed by a Parquet file. The Parquet file is loaded with Pandas, and paths are resolved relative to the directory containing that Parquet file.

Expected inputs:

- `--dataset_fname /path/to/dataset.parquet`
- An ontology file in the same directory, named `ontology.json` by default
- Optionally, a Parquet attribute named `ontology_fname` pointing to a different ontology filename

Typical columns:

| Modality | Required columns |
| --- | --- |
| Image segmentation | `image`, `label`, `split` |
| LiDAR segmentation | `points`, `label`, `split` |

## Generic

Generic datasets are useful when files can be paired by matching wildcard captures in input and label patterns. They support image and LiDAR segmentation.

Expected inputs:

- At least one of `--train_dataset_dir`, `--val_dataset_dir`, or `--test_dataset_dir`
- `--data_suffix`
- `--label_suffix`
- `--dataset_ontology`

The data and label suffixes must contain the same number of `*` wildcards. For example:

```shell
--data_suffix "*_image.png" --label_suffix "*_label.png"
```

The ontology may be either a list of class names:

```json
["background", "road", "vegetation"]
```

or a dictionary:

```json
{
  "background": {"idx": 0, "rgb": [0, 0, 0]},
  "road": {"idx": 1, "rgb": [128, 64, 128]}
}
```

## GOOSE

GOOSE supports image and LiDAR semantic segmentation. The adapter expects the official split directory layout and reads `goose_label_mapping.csv` from the first provided split root.

Expected inputs:

- One or more of `--train_dataset_dir`, `--val_dataset_dir`, `--test_dataset_dir`
- Each split root should contain `goose_label_mapping.csv`
- Image data under `images/<split>/*/*_windshield_vis.png`
- Image labels under `labels/<split>/<scene>/*_labelids.png`
- LiDAR data under `lidar/<split>/*/*_vls128.bin` for GOOSE or `*_pcl.bin` for GOOSE Ex
- LiDAR labels under `labels/<split>/<scene>/*_goose.label`

## RELLIS-3D

RELLIS-3D supports image and LiDAR semantic segmentation. The adapter uses official `.lst` split files and a YAML ontology file.

Expected inputs:

- `--dataset_dir`: directory containing the extracted data and labels
- `--split_dir`: directory containing split files
- `--dataset_ontology`: RELLIS-3D ontology YAML

Image split files:

- `train.lst`
- `val.lst`
- `test.lst`

LiDAR split files:

- `pt_train.lst`
- `pt_val.lst`
- `pt_test.lst`

Each row in a split file is expected to contain the relative data path and relative label path separated by a space.

## RUGD

RUGD supports image semantic segmentation. Labels are RGB masks, so the adapter initializes the base image segmentation dataset with RGB-label handling enabled.

Expected inputs:

- `--images_dir`
- `--labels_dir`
- `--dataset_ontology`, usually `RUGD_annotation-colormap.txt`

The default train, validation, and test split assignment is built into the adapter using the sequence names from the RUGD paper.

## WildScenes

WildScenes supports image and LiDAR semantic segmentation through library adapters. The adapters expect official CSV split files and use ontology definitions embedded in the adapter source.

Expected inputs in Python:

- `dataset_dir`: root of the WildScenes data
- `split_dir`: directory containing `train.csv`, `val.csv`, and `test.csv`

Use the 2D split files for `WildscenesImageSegmentationDataset` and the 3D split files for `WildscenesLiDARSegmentationDataset`.

## Cityscapes

Cityscapes supports image semantic segmentation through the Python API.

Expected inputs in Python:

- One or more of `train_dataset_root`, `val_dataset_root`, `test_dataset_root`
- Images under `leftImg8bit_trainvaltest/leftImg8bit/<split>/<city>/`
- Labels under `gtFine/<split>/<city>/`
- Default image suffix: `_leftImg8bit.png`
- Default label suffix: `_gtFine_labelIds.png`

The adapter can build either Cityscapes label-id ontologies or train-id ontologies. When using train IDs, provide train-id labels with `label_suffix="_gtFine_labelTrainIds.png"`.

## COCO

COCO supports image object detection and is available from the CLI and Python API.

Expected layout:

```text
dataset_root/
  images/
    train2017/
    val2017/
  annotations/
    instances_train2017.json
    instances_val2017.json
```

Expected CLI inputs:

- `--dataset_format coco`
- `--dataset_dir /path/to/dataset_root`
- `--split train` or `--split val`

The CLI currently supports one COCO split at a time. The adapter looks for an image directory matching the split name, such as `val2017`, and an annotation file matching `instances_<split>*.json`.

## YOLO

YOLO supports image object detection through the Python API. The adapter reads an Ultralytics-style dataset YAML file.

Expected YAML fields:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test
names:
  0: person
  1: vehicle
```

The adapter expects labels in matching `labels/<split>` directories and converts YOLO normalized center-width-height boxes into absolute `[x1, y1, x2, y2]` boxes.

## nuImages

nuImages supports image object detection and image semantic segmentation through the Python API.

Expected inputs in Python:

- `dataset_dir`: nuImages root directory
- `version`, defaulting to `v1.0-mini`
- `split`, defaulting to `train`



