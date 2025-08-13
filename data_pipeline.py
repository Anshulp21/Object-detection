"""
Data Pipeline for IDD-Lite -> YOLOv5
- Scans IDD-Lite structure (leftImg8bit, gtFine)
- Converts instance masks to YOLOv5 bbox labels into labels_yolo/{train,val}
- Generates data YAML with discovered classes
"""
from __future__ import annotations
import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from collections import OrderedDict, defaultdict

import yaml
import numpy as np
import cv2


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
YOLO_DATA_ROOT = PROJECT_ROOT / "yolo_data"
IMAGES_LINK_DIR = YOLO_DATA_ROOT / "images"
LABELS_DIR = YOLO_DATA_ROOT / "labels"
DATA_DIR.mkdir(exist_ok=True)
IMAGES_LINK_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)


def _symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        # Windows symlink for files needs admin; fall back to copy
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def discoverSplits(iddRoot: Path) -> Dict[str, Path]:
    left = iddRoot / "leftImg8bit"
    gt = iddRoot / "gtFine"
    assert left.exists() and gt.exists(), f"Expected folders 'leftImg8bit' and 'gtFine' under {iddRoot}"
    splits = {}
    for split in ["train", "val"]:
        if (left / split).exists() and (gt / split).exists():
            splits[split] = left / split
    assert splits, f"No train/val splits found under {left}"
    return {k: v for k, v in sorted(splits.items())}


def scanOneSplit(leftSplitDir: Path) -> List[Path]:
    # Collect image files recursively
    imgs: List[Path] = []
    for p in leftSplitDir.rglob("*.png"):
        imgs.append(p)
    for p in leftSplitDir.rglob("*.jpg"):
        imgs.append(p)
    return imgs


def buildLabelPath(imgPath: Path, iddRoot: Path) -> Tuple[Path, Path]:
    # Find corresponding label and inst_label under gtFine with similar relative path and filename prefix
    # Example: leftImg8bit/train/<city>/<file>.png -> gtFine/train/<city>/<file>_{label,inst_label}.png
    rel = imgPath.relative_to(iddRoot / "leftImg8bit")
    split = rel.parts[0]
    city = rel.parts[1]
    stem = imgPath.stem  # e.g., '024541_image' or 'frame0001'
    # IDD-Lite uses '*_image.jpg' for RGB, but labels are '*_label.png' without the '_image' part.
    base = stem[:-6] if stem.endswith("_image") else stem
    label = iddRoot / "gtFine" / split / city / f"{base}_label.png"
    inst = iddRoot / "gtFine" / split / city / f"{base}_inst_label.png"
    return label, inst


def convertDataset(iddRoot: Path) -> Tuple[Path, Path, List[str]]:
    """Convert IDD-Lite masks to YOLOv5 txt labels and mirror images.
    Returns (images_root, labels_root, class_names)
    """
    splits = discoverSplits(iddRoot)

    # Discover classes dynamically by scanning label masks for unique values

    value_to_index: Dict[int, int] = OrderedDict()
    fallback_mode = False

    for split, left_dir in splits.items():
        imgs = scanOneSplit(left_dir)
        for imgPath in imgs[:500]:  # sample more files to improve discovery
            labelPath, _ = buildLabelPath(imgPath, iddRoot)
            if not labelPath.exists():
                continue
            m = cv2.imread(str(labelPath), cv2.IMREAD_UNCHANGED)
            if m is None:
                continue
            # If label image has 3 channels (rare), convert to grayscale
            if len(m.shape) == 3 and m.shape[2] >= 3:
                try:
                    m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
                except Exception:
                    # Fallback: take first channel
                    m = m[:, :, 0]
            uniques = np.unique(m)
            for v in uniques.tolist():
                if v in (0, 255):  # skip background/void commonly used
                    continue
                if v not in value_to_index:
                    value_to_index[v] = len(value_to_index)

    if not value_to_index:
        # Diagnostics: collect some unique values and write to file
        print("[WARN] No classes discovered. Enabling fallback single-class mode. Writing diagnostics to data/diag_values.json", flush=True)
        diag = defaultdict(list)
        diag_count = 0
        for split, left_dir in splits.items():
            imgs = scanOneSplit(left_dir)
            for imgPath in imgs[:10]:
                labelPath, _ = buildLabelPath(imgPath, iddRoot)
                if not labelPath.exists():
                    continue
                m = cv2.imread(str(labelPath), cv2.IMREAD_UNCHANGED)
                if m is None:
                    continue
                if len(m.shape) == 3 and m.shape[2] >= 3:
                    try:
                        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
                    except Exception:
                        m = m[:, :, 0]
                u = np.unique(m)
                diag[str(labelPath)].extend(u[:20].astype(int).tolist())
                diag_count += 1
                if diag_count >= 10:
                    break
            if diag_count >= 10:
                break
        (DATA_DIR).mkdir(exist_ok=True)
        with (DATA_DIR / "diag_values.json").open("w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2)
        # Fallback: single class 'object'
        value_to_index = {1: 0}
        fallback_mode = True

    # IDD-Lite class mapping based on common segmentation values
    IDD_CLASS_MAPPING = {
        1: "road", 2: "drivable_fallback", 3: "sidewalk", 4: "non_drivable_fallback",
        5: "person", 6: "rider", 7: "motorcycle", 8: "bicycle", 9: "autorickshaw",
        10: "car", 11: "truck", 12: "bus", 13: "vehicle_fallback", 14: "curb",
        15: "wall", 16: "fence", 17: "guard_rail", 18: "billboard", 19: "traffic_sign",
        20: "traffic_light", 21: "pole", 22: "obs_str_obj_fallback", 23: "building",
        24: "bridge", 25: "vegetation", 26: "sky", 27: "fallback_background"
    }
    
    if fallback_mode:
        class_names = ["object"]
    else:
        class_names = []
        for v in value_to_index.keys():
            class_name = IDD_CLASS_MAPPING.get(v, f"class_{v}")
            class_names.append(class_name)

    # Save classes file
    classes_file = DATA_DIR / "idd_lite_classes.json"
    with classes_file.open("w", encoding="utf-8") as f:
        json.dump({"mapping": value_to_index, "names": class_names}, f, indent=2)

    # Create YOLO directories
    for split in splits.keys():
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)
        (IMAGES_LINK_DIR / split).mkdir(parents=True, exist_ok=True)

    # Convert annotations: instance mask -> bbox per instance

    for split, left_dir in splits.items():
        imgs = scanOneSplit(left_dir)
        for imgPath in imgs:
            labelPath, instPath = buildLabelPath(imgPath, iddRoot)
            if not (labelPath.exists() and instPath.exists()):
                # Skip if annotation missing
                continue
            labelImg = cv2.imread(str(labelPath), cv2.IMREAD_UNCHANGED)
            instImg = cv2.imread(str(instPath), cv2.IMREAD_UNCHANGED)
            if labelImg is None or instImg is None:
                continue
            h, w = labelImg.shape[:2]
            instances = np.unique(instImg)
            yolo_lines: List[str] = []
            for ins_id in instances.tolist():
                if ins_id == 0:
                    continue
                mask = (instImg == ins_id)
                if not mask.any():
                    continue
                # Determine class by majority label within the instance mask
                cls_idx = 0  # default for fallback
                if not fallback_mode:
                    clsVals, counts = np.unique(labelImg[mask], return_counts=True)
                    # Exclude void/background ids commonly 0 or 255
                    filtered = [(cv, ct) for cv, ct in zip(clsVals.tolist(), counts.tolist()) if cv not in (0, 255)]
                    if not filtered:
                        continue
                    clsVal = max(filtered, key=lambda x: x[1])[0]
                    if clsVal not in value_to_index:
                        # Skip rare/unseen class
                        continue
                    clsIdx = value_to_index[clsVal]
                ys, xs = np.where(mask)
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                # Convert to YOLO xywh normalized
                xc = (x_min + x_max) / 2.0 / w
                yc = (y_min + y_max) / 2.0 / h
                bw = (x_max - x_min) / w
                bh = (y_max - y_min) / h
                if bw <= 0 or bh <= 0:
                    continue
                yolo_lines.append(f"{clsIdx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            # Write label file only if any boxes
            if yolo_lines:
                rel = imgPath.relative_to(iddRoot / "leftImg8bit")
                txt_out = (LABELS_DIR / split / rel).with_suffix(".txt")
                txt_out.parent.mkdir(parents=True, exist_ok=True)
                with txt_out.open("w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_lines))
                # Mirror/copy image alongside into images_yolo/
                img_out = IMAGES_LINK_DIR / split / rel
                _symlink_or_copy(imgPath, img_out)

    # Create data YAML (YOLOv5 expects labels directory parallel to images and auto-resolves by replacing 'images' with 'labels')
    data_yaml = DATA_DIR / "idd_lite.yaml"
    yaml_dict = {
        "path": str(PROJECT_ROOT),
        "train": str(IMAGES_LINK_DIR / "train"),
        "val": str(IMAGES_LINK_DIR / "val"),
        "names": class_names,
    }
    with data_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False, allow_unicode=True)

    return IMAGES_LINK_DIR, LABELS_DIR, class_names


def prepareDataset(iddRoot: Path) -> Path:
    """High-level API: ensure conversion done and return path to data YAML."""
    data_yaml = DATA_DIR / "idd_lite.yaml"
    needs_convert = not data_yaml.exists()
    # Also convert if images folders are missing or empty
    train_imgs_dir = IMAGES_LINK_DIR / "train"
    val_imgs_dir = IMAGES_LINK_DIR / "val"
    if not needs_convert:
        # Check presence of at least one image in each split (png/jpg)
        def has_images(d: Path) -> bool:
            return d.exists() and (any(d.rglob("*.png")) or any(d.rglob("*.jpg")))

        if not has_images(train_imgs_dir) or not has_images(val_imgs_dir):
            print("[INFO] YOLO images directory empty or missing. Re-generating dataset...", flush=True)
            needs_convert = True

    if needs_convert:
        convertDataset(iddRoot)
    return data_yaml


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare IDD-Lite dataset for YOLOv5")
    DEFAULT_IDD_ROOT = Path(__file__).resolve().parent / "idd-lite" / "idd20k_lite"
    parser.add_argument("--idd_root", type=str, required=False, default=str(DEFAULT_IDD_ROOT), help="Path to idd20k_lite root (containing leftImg8bit, gtFine)")
    args = parser.parse_args()
    yml = prepareDataset(Path(args.idd_root))
    print(f"Data YAML ready at: {yml}")
