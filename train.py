# ================================================================
# TOMATO DETECTION - KAGGLE NOTEBOOK TRAINER
# Run each cell block separately in Kaggle
# ================================================================

# ── CELL 1: Install & imports ────────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "ultralytics", "-q"])

import os, json, shutil, yaml, torch
from pathlib import Path
from ultralytics import YOLO
print(f"PyTorch  : {torch.__version__}")
print(f"GPU      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"CUDA     : {torch.version.cuda}")


# ── CELL 2: Upload & extract your dataset ───────────────────────
# Option A: If you uploaded combined_dataset.zip to Kaggle
# import zipfile
# with zipfile.ZipFile("/kaggle/input/YOUR_DATASET/combined_dataset.zip") as z:
#     z.extractall("/kaggle/working/dataset")

# Option B: Use the tomato-bunches dataset directly from Kaggle
# Search "tomato bunches red green" in Kaggle Datasets and add it
# It will appear at /kaggle/input/tomato-bunches-red-green/

COCO_ROOT   = "/kaggle/input/tomato-bunches-red-green"   # Kaggle dataset path
OUTPUT_ROOT = "/kaggle/working/combined_dataset"


# ── CELL 3: Convert COCO -> YOLO ────────────────────────────────
COCO_CLASS_MAP = {1: 2, 2: 3}   # red->Ripe(2)  green->Unripe(3)
CLASS_NAMES    = ["Damaged", "Old", "Ripe", "Unripe"]
IMG_EXTS       = {".jpg", ".jpeg", ".png", ".bmp"}

def coco_to_yolo_boxes(annotations, img_w, img_h):
    lines = []
    for ann in annotations:
        yolo_cls = COCO_CLASS_MAP.get(ann["category_id"])
        if yolo_cls is None:
            continue
        x, y, bw, bh = ann["bbox"]
        cx = min(max((x + bw/2) / img_w, 0), 1)
        cy = min(max((y + bh/2) / img_h, 0), 1)
        nw = min(max(bw / img_w, 0), 1)
        nh = min(max(bh / img_h, 0), 1)
        lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines

def convert_split(split):
    # Try both annotation locations
    for ann_dir in ["annotations", "."]:
        json_path = Path(COCO_ROOT) / ann_dir / f"{split}_instances.json"
        if json_path.exists():
            break
    else:
        print(f"  [{split}] annotation file not found, skipping")
        return

    img_src = Path(COCO_ROOT) / split
    img_out = Path(OUTPUT_ROOT) / split / "images"
    lbl_out = Path(OUTPUT_ROOT) / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    ann_map = {}
    for ann in data["annotations"]:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    copied = labeled = 0
    for img_info in data["images"]:
        src = img_src / img_info["file_name"]
        if not src.exists():
            continue
        shutil.copy2(src, img_out / img_info["file_name"])
        copied += 1
        lines = coco_to_yolo_boxes(
            ann_map.get(img_info["id"], []),
            img_info["width"], img_info["height"])
        (lbl_out / (Path(img_info["file_name"]).stem + ".txt")).write_text("\n".join(lines))
        if lines: labeled += 1

    print(f"  [{split}] images: {copied}  labeled: {labeled}")

print("Converting dataset...")
for split in ["train", "val", "test"]:
    convert_split(split)

# Write dataset.yaml
yaml_cfg = {
    "path":  OUTPUT_ROOT,
    "train": "train/images",
    "val":   "val/images",
    "test":  "test/images",
    "nc":    len(CLASS_NAMES),
    "names": CLASS_NAMES,
}
yaml_path = Path(OUTPUT_ROOT) / "dataset.yaml"
yaml_path.write_text(yaml.dump(yaml_cfg, sort_keys=False))
print(f"dataset.yaml -> {yaml_path}")

# Verify counts
for split in ["train", "val"]:
    imgs = list((Path(OUTPUT_ROOT) / split / "images").glob("*"))
    lbls = list((Path(OUTPUT_ROOT) / split / "labels").glob("*.txt"))
    print(f"  {split}: {len(imgs)} images  {len(lbls)} labels")


# ── CELL 4: TRAIN ───────────────────────────────────────────────
model = YOLO("yolov8s.pt")

results = model.train(
    data        = str(yaml_path),
    epochs      = 80,
    imgsz       = 640,
    batch       = 32,        # Kaggle T4 has 16GB, can handle 32
    device      = "0",       # GPU
    project     = "/kaggle/working/runs",
    name        = "tomato_v3",
    patience    = 20,
    half        = True,      # FP16 - faster
    save        = True,
    val         = True,
    plots       = True,
    exist_ok    = True,
    workers     = 2,         # Kaggle has limited CPU cores
    # Augmentation
    hsv_h=0.015, hsv_s=0.7,  hsv_v=0.4,
    degrees=10,  translate=0.1, scale=0.5,
    flipud=0.1,  fliplr=0.5,
    mosaic=1.0,  mixup=0.15, copy_paste=0.3,
)

best = Path("/kaggle/working/runs/tomato_v3/weights/best.pt")
print(f"\nBest weights: {best}")
print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")


# ── CELL 5: TEST on sample image ────────────────────────────────
import cv2
from matplotlib import pyplot as plt

model_best = YOLO(str(best))

# Test on a val image
test_imgs = list((Path(OUTPUT_ROOT) / "val" / "images").glob("*.jpg"))
if test_imgs:
    test_img = str(test_imgs[0])
    res = model_best.predict(test_img, conf=0.25, iou=0.40, imgsz=640)
    annotated = res[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_rgb)
    plt.axis("off")
    plt.title("Sample Detection Result")
    plt.show()

    # Count per class
    boxes = res[0].boxes
    if boxes is not None:
        from collections import Counter
        cls_counts = Counter([CLASS_NAMES[int(c)] for c in boxes.cls])
        print("\nDetections:")
        for cls, cnt in cls_counts.items():
            print(f"  {cls:<12}: {cnt}")
        print(f"  {'TOTAL':<12}: {sum(cls_counts.values())}")


# ── CELL 6: Download weights ─────────────────────────────────────
# In Kaggle, go to the right panel -> Output -> download best.pt
# Then copy it to: C:/code/Tomato/runs/tomato_v3/weights/best.pt
print(f"\nDownload your weights from Kaggle Output panel:")
print(f"  /kaggle/working/runs/tomato_v3/weights/best.pt")
print(f"\nThen in display.py set:")
print(f'  WEIGHTS = "C:/code/Tomato/runs/tomato_v3/weights/best.pt"')