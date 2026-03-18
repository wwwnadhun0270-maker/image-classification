# coco_to_yolo.py
# Converts tomato-bunches COCO dataset -> YOLO format
# Merges with original tomato frout dataset
# Final classes: Damaged=0  Old=1  Ripe=2  Unripe=3

import json
import shutil
import subprocess
import sys
from pathlib import Path

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
try:
    import yaml
except ImportError:
    print("Installing pyyaml...")
    install("pyyaml")
    import yaml


# ════════════════════════════════════════════════════════════════
# SETTINGS
# ════════════════════════════════════════════════════════════════
COCO_ROOT = "C:/code/Tomato/datasets/tomato-bunches-red-green"
OLD_DATASET = "C:/code/Tomato/tomato frout"
OUTPUT_ROOT = "C:/code/Tomato/combined_dataset"
# ════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════

COCO_CLASS_MAP = {1: 2, 2: 3}   # red bunch->Ripe(2)  green bunch->Unripe(3)
OLD_CLASS_MAP  = {"Damaged": 0, "Old": 1, "Ripe": 2, "Unripe": 3}
CLASS_NAMES    = ["Damaged", "Old", "Ripe", "Unripe"]
IMG_EXTS       = {".jpg", ".jpeg", ".png", ".jfif", ".bmp"}


def coco_to_yolo_boxes(annotations, img_w, img_h):
    lines = []
    for ann in annotations:
        yolo_cls = COCO_CLASS_MAP.get(ann["category_id"])
        if yolo_cls is None:
            continue
        x, y, bw, bh = ann["bbox"]
        cx = min(max((x + bw / 2) / img_w, 0), 1)
        cy = min(max((y + bh / 2) / img_h, 0), 1)
        nw = min(max(bw / img_w, 0), 1)
        nh = min(max(bh / img_h, 0), 1)
        lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def convert_coco_split(split):
    json_path = None
    for ann_dir in ["annotations", "."]:
        p = Path(COCO_ROOT) / ann_dir / f"{split}_instances.json"
        if p.exists():
            json_path = p
            break
    if json_path is None:
        print(f"  [{split}] annotation not found, skipping")
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
        if lines:
            labeled += 1

    print(f"  [{split}] images: {copied}  labeled: {labeled}")


def merge_old_dataset():
    old = Path(OLD_DATASET)
    for split in ["train", "val"]:
        split_dir = old / split
        if not split_dir.exists():
            continue
        img_out = Path(OUTPUT_ROOT) / split / "images"
        lbl_out = Path(OUTPUT_ROOT) / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        count = 0
        for cls_name, cls_id in OLD_CLASS_MAP.items():
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue
                lbl_src = img_path.with_suffix(".txt")
                if not lbl_src.exists():
                    continue
                # Unique name to avoid collisions
                dst_name = f"frout_{cls_name}_{img_path.name}"
                shutil.copy2(img_path, img_out / dst_name)
                new_lines = []
                for line in lbl_src.read_text().strip().splitlines():
                    parts = line.split()
                    if parts:
                        parts[0] = str(cls_id)
                        new_lines.append(" ".join(parts))
                (lbl_out / f"frout_{cls_name}_{img_path.stem}.txt").write_text("\n".join(new_lines))
                count += 1
        print(f"  [old/{split}] merged {count} images")


def write_yaml():
    cfg = {
        "path":  OUTPUT_ROOT,
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    p = Path(OUTPUT_ROOT) / "dataset.yaml"
    p.write_text(yaml.dump(cfg, sort_keys=False))
    print(f"\n  dataset.yaml -> {p}")


def main():
    print("=" * 55)
    print("  COCO -> YOLO CONVERTER + DATASET MERGER")
    print("  Damaged=0  Old=1  Ripe=2  Unripe=3")
    print("=" * 55)

    if not Path(COCO_ROOT).exists():
        print(f"\n  COCO dataset not found at:\n  {COCO_ROOT}")
        print("  Check the path and try again.")
        return

    print("\n  Converting tomato-bunches dataset...")
    for split in ["train", "val", "test"]:
        convert_coco_split(split)

    if Path(OLD_DATASET).exists():
        print("\n  Merging original tomato frout dataset...")
        merge_old_dataset()
    else:
        print(f"\n  Old dataset not found at {OLD_DATASET}, skipping merge")

    write_yaml()

    print("\n  Final dataset summary:")
    for split in ["train", "val"]:
        imgs = list((Path(OUTPUT_ROOT) / split / "images").glob("*"))
        lbls = list((Path(OUTPUT_ROOT) / split / "labels").glob("*.txt"))
        print(f"  {split}: {len(imgs)} images  |  {len(lbls)} labels")

    print("\n  Done! Now run train.py")


if __name__ == "__main__":
    main()
