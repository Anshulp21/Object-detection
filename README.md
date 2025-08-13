# Object Detection on IDD-Lite with YOLOv5 (Python 3.11)

End-to-end pipeline to prepare IDD-Lite, train YOLOv5, and run inference on Windows or Google Colab.
All defaults are hardcoded so you can run: `python main.py`.

Key files:
- `main.py` — Orchestrates dataset prep + training (1 epoch by default).
- `data_pipeline.py` — Converts IDD-Lite masks to YOLO labels and generates `yolo_data/` and `data/idd_lite.yaml`.
- `train.py` — Wraps YOLOv5 training/validation with sensible defaults.
- `data/idd_lite.yaml` — Auto-generated dataset YAML.
- `runs_idd/` — Training and detection outputs.

---

## 2) Setup (Windows)
```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

---


---

## 4) Run the full pipeline (prep + train 1 epoch)
```powershell
python main.py
```
What happens:
- Convert masks to YOLO labels under `yolo_data/` (keeps subfolders per city).
- Create `data/idd_lite.yaml` with the 10 fixed class names.
- Train YOLOv5 for 1 epoch and save weights to `runs_idd/exp/weights/best.pt`.

Override (optional):
```powershell
python main.py --device 0             # use GPU 0
python main.py --idd_root D:\idd20k_lite  # custom dataset root
```

---

## 5) Detect (inference)
Because images are nested in subfolders, use a recursive glob.

- Detect on all validation images:
```powershell
yolov5 detect --weights runs_idd\exp\weights\best.pt --source "yolo_data\images\val\**" --img 640 --conf-thres 0.25 --device cpu --project runs_idd --name detect_val --exist-ok
```

- Detect on all training images:
```powershell
yolov5 detect --weights runs_idd\exp\weights\best.pt --source "yolo_data\images\train\**" --img 640 --conf-thres 0.25 --device cpu --project runs_idd --name detect_train --exist-ok
```

- Single image:
```powershell
yolov5 detect --weights runs_idd\exp\weights\best.pt --source "path\to\image.jpg" --img 640 --conf-thres 0.25 --device cpu --project runs_idd --name detect_single --exist-ok
```

Outputs are saved under `runs_idd/<name>/`.





# How to Run

```bash
1.python main.py


2.yolov5 detect --weights runs_idd\exp\weights\best.pt --source "yolo_data\images\val\**" --img 640 --conf-thres 0.25 --device cpu --project runs_idd --name detect_val --exist-ok
