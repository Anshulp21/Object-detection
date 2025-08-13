"""
Main script: IDD-Lite dataset preparation + YOLOv5 training
- Step 1: Runs data_pipeline.prepareDataset() to prepare the dataset
- Step 2: Starts training using train.trainYolov5()
"""
from __future__ import annotations
from pathlib import Path
import argparse

from data_pipeline import prepareDataset
from train import trainYolov5


def main():
    # Default dataset root inside this repo
    DEFAULT_IDD_ROOT = Path(__file__).resolve().parent / "idd-lite" / "idd20k_lite"

    p = argparse.ArgumentParser(description="YOLOv5 training on IDD-Lite dataset")
    p.add_argument(
        "--idd_root",
        type=str,
        required=False,
        default=str(DEFAULT_IDD_ROOT),
        help="Root path to idd20k_lite dataset (containing leftImg8bit and gtFine folders)",
    )
    p.add_argument("--weights", type=str, default="yolov5s.pt", help="initial weights, e.g., yolov5s.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="cpu", help="Device to use: 'cpu' or '0' for GPU if available")
    p.add_argument("--project", type=str, default="runs_idd")
    p.add_argument("--name", type=str, default="exp")
    args = p.parse_args()

    iddRoot = Path(args.idd_root).resolve()
    dataYaml = prepareDataset(iddRoot)
    print(f"[INFO] Data YAML: {dataYaml}")

    rc = trainYolov5(
        dataYaml=dataYaml,
        weights=args.weights,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
