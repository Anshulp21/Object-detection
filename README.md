# Object Detection on IDD-Lite with YOLOv5 (Python 3.11)

End-to-end pipeline to prepare **IDD-Lite**, train **YOLOv5**, and run inference on **Windows** or 
All defaults are hardcoded, so you can run the full process with just:

```bash
python main.py
```

---

## 📂 Key Files

- **`main.py`** — Orchestrates dataset preparation + YOLOv5 training (default: 1 epoch).
- **`data_pipeline.py`** — Converts IDD-Lite masks to YOLO labels, generates `yolo_data/` and `data/idd_lite.yaml`.
- **`train.py`** — Wraps YOLOv5 training/validation with sensible defaults.
- **`data/idd_lite.yaml`** — Auto-generated dataset config.
- **`runs_idd/`** — Training and detection outputs.

---

## ⚙️ Setup (Windows)

1. **Clone the repository**
```powershell
git clone https://github.com/Anshulp21/Object-detection.git
cd Object-detection
```

2. **Create and activate virtual environment**
```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
```

3. **Install dependencies**
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---



```powershell
python main.py
```

What this does:
- Converts masks → YOLO labels (`yolo_data/`), keeping city-based subfolders.
- Creates `data/idd_lite.yaml` with the 10 fixed class names.
- Saves weights to:  
  ```
  runs_idd/exp/weights/best.pt
  ```

Optional overrides:
```powershell
python main.py --device 0                 # Use GPU 0
python main.py --idd_root D:\idd20k_lite  # Custom dataset root
```

---

## 🔍 Detect (Inference)

### Detect on **all validation images**
```powershell
yolov5 detect --weights runs_idd\exp\weights\best.pt --source "yolo_data\images\val\**" --img 640 --conf-thres 0.25 --device cpu --project runs_idd --name detect_val --exist-ok
```

### Detect on **all training images**
```powershell
yolov5 detect --weights runs_idd\exp\weights\best.pt --source "yolo_data\images\train\**" --img 640 --conf-thres 0.25 --device cpu --project runs_idd --name detect_train --exist-ok
```

### Detect on a **single image**
```powershell
yolov5 detect --weights runs_idd\exp\weights\best.pt --source "path\to\image.jpg" --img 640 --conf-thres 0.25 --device cpu --project runs_idd --name detect_single --exist-ok
```

All outputs are saved under:
```
runs_idd/<name>/
```

---

## 🏃 Quick Run Commands

```bash
1. python main.py

2. yolov5 detect --weights runs_idd\exp\weights\best.pt --source "yolo_data\images\val\**" --img 640 --conf-thres 0.25 --device cpu --project runs_idd --name detect_val --exist-ok
```

---


