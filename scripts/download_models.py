"""
scripts/download_models.py
Downloads the base YOLO weights into TrafficAI/models/.
Run once before starting the API: python scripts/download_models.py
"""

from pathlib import Path
from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Ultralytics auto-downloads to the current working directory the first
# time you instantiate YOLO("yolo11s.pt"), so we point it explicitly at
# models/ and then move it if needed.
TARGET = MODELS_DIR / "yolo11s.pt"

if TARGET.exists():
    print(f"Already present: {TARGET}")
else:
    print("Downloading yolo11s.pt (general-purpose, 21MB)...")
    model = YOLO("yolo11s.pt")  # triggers download to ./yolo11s.pt
    downloaded = Path("yolo11s.pt")
    if downloaded.exists():
        downloaded.rename(TARGET)
    print(f"Saved to {TARGET}")

print(
    "\nNOTE: yolo11s.pt is the stock Ultralytics COCO model — it detects "
    "general vehicle classes (car, bus, truck, motorcycle, bicycle) but is "
    "NOT specifically fine-tuned for Indian traffic (autos, e-rickshaws, "
    "overloaded trucks, non-standard buses, etc). For real Indian-vehicle "
    "accuracy you'll need to fine-tune on an Indian dataset (e.g. IDD - "
    "Indian Driving Dataset) — see the bottom of the chat response for how."
)