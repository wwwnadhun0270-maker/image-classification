# display.py - Tomato Detector (Damaged / Old / Ripe / Unripe)

import cv2
import torch
import time
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# === CHANGE THESE ===
WEIGHTS = "C:/code/Tomato/runs/tomato_v3/weights/best.pt"
SOURCE  = "C:/Users/Nadhu/Downloads/4688-179650983_medium.mp4"
SAVE    = False
# ====================

CLASS_NAMES = ["Damaged", "Old", "Ripe", "Unripe"]
CONF        = 0.25
IOU         = 0.40
IMG_SIZE    = 640
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp"}
OUTPUT_DIR  = Path("C:/code/Tomato/tomato_outputs")

CLASS_COLORS = {
    "Damaged": (0,   40,  210),   # red
    "Old":     (0,  140,  255),   # orange
    "Ripe":    (30, 200,   30),   # green
    "Unripe":  (20, 180,  180),   # yellow-green
}
FONT = cv2.FONT_HERSHEY_SIMPLEX
BOLD = cv2.FONT_HERSHEY_DUPLEX


def draw_boxes(frame, results):
    counts = defaultdict(int)
    boxes  = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return frame, counts
    for box in boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        name   = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        color  = CLASS_COLORS.get(name, (255, 200, 0))
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Corner accents
        c, t = 14, 4
        for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (px, py), (px + dx*c, py), color, t)
            cv2.line(frame, (px, py), (px, py + dy*c), color, t)

        # Label
        label = f" {name} {conf:.0%} "
        (lw, lh), bl = cv2.getTextSize(label, FONT, 0.52, 1)
        ly = max(y1 - lh - 6, 0)
        cv2.rectangle(frame, (x1, ly), (x1 + lw, ly + lh + 6), color, -1)
        cv2.putText(frame, label, (x1, ly + lh + 2),
                    FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        counts[name] += 1
    return frame, counts


def draw_panel(frame, counts, fps=0.0):
    h, w  = frame.shape[:2]
    total = sum(counts.values())
    pw    = 210
    ph    = 50 + len(CLASS_NAMES) * 30 + 50
    px    = w - pw - 12
    py    = 12

    # Background
    ov = frame.copy()
    cv2.rectangle(ov, (px-8, py-8), (px+pw, py+ph), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (px-8, py-8), (px+pw, py+ph), (70, 70, 70), 1)

    # Title
    cv2.putText(frame, "TOMATO COUNTER", (px, py+18),
                BOLD, 0.52, (200, 200, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (px-8, py+26), (px+pw, py+26), (60, 60, 60), 1)

    # Per class
    y = py + 50
    for cls in CLASS_NAMES:
        cnt   = counts.get(cls, 0)
        color = CLASS_COLORS.get(cls, (180, 180, 180))
        cv2.circle(frame, (px+8, y-5), 6, color, -1)
        cv2.putText(frame, cls, (px+22, y),
                    FONT, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
        s = str(cnt)
        (sw, _), _ = cv2.getTextSize(s, BOLD, 0.58, 1)
        cv2.putText(frame, s, (px+pw-sw-6, y),
                    BOLD, 0.58, color if cnt > 0 else (60, 60, 60),
                    1, cv2.LINE_AA)
        y += 30

    # Total
    cv2.line(frame, (px-8, y-4), (px+pw, y-4), (60, 60, 60), 1)
    cv2.putText(frame, "TOTAL", (px, y+22),
                BOLD, 0.60, (255, 255, 140), 1, cv2.LINE_AA)
    ts = str(total)
    (tw, _), _ = cv2.getTextSize(ts, BOLD, 0.80, 2)
    cv2.putText(frame, ts, (px+pw-tw-6, y+24),
                BOLD, 0.80, (80, 255, 80), 2, cv2.LINE_AA)

    if fps > 0:
        cv2.putText(frame, f"FPS {fps:.1f}", (px, py+ph-4),
                    FONT, 0.38, (90, 90, 90), 1, cv2.LINE_AA)
    return frame


def predict(model, frame):
    results = model.predict(frame, conf=CONF, iou=IOU, imgsz=IMG_SIZE, verbose=False)
    return draw_boxes(frame, results)


def run_image(model, img_path):
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"Cannot read: {img_path}"); return
    frame, counts = predict(model, frame)
    frame = draw_panel(frame, counts)
    print(f"\n{img_path.name}")
    for c in CLASS_NAMES:
        print(f"  {c:<10}: {counts.get(c,0)}")
    print(f"  {'TOTAL':<10}: {sum(counts.values())}")
    if SAVE:
        OUTPUT_DIR.mkdir(exist_ok=True)
        cv2.imwrite(str(OUTPUT_DIR / img_path.name), frame)
    cv2.namedWindow("Tomato Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tomato Detection", 960, 640)
    cv2.imshow("Tomato Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_folder(model, folder):
    imgs = sorted([f for f in folder.iterdir() if f.suffix.lower() in IMG_EXTS])
    if not imgs:
        print(f"No images in {folder}"); return
    grand = defaultdict(int)
    print(f"Processing {len(imgs)} images — press Q to stop")
    for p in imgs:
        frame = cv2.imread(str(p))
        if frame is None: continue
        frame, counts = predict(model, frame)
        frame = draw_panel(frame, counts)
        for c, n in counts.items(): grand[c] += n
        if SAVE:
            OUTPUT_DIR.mkdir(exist_ok=True)
            cv2.imwrite(str(OUTPUT_DIR / p.name), frame)
        cv2.namedWindow("Tomato Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tomato Detection", 960, 640)
        cv2.imshow("Tomato Detection", frame)
        if cv2.waitKey(600) in (ord("q"), 27): break
    cv2.destroyAllWindows()
    print("\nTotal:")
    for c in CLASS_NAMES: print(f"  {c:<10}: {grand.get(c,0)}")
    print(f"  {'TOTAL':<10}: {sum(grand.values())}")


def run_stream(model, source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open: {source}"); return
    writer = None
    if SAVE:
        OUTPUT_DIR.mkdir(exist_ok=True)
        nm = "webcam.mp4" if isinstance(source, int) else Path(str(source)).stem+"_det.mp4"
        writer = cv2.VideoWriter(str(OUTPUT_DIR/nm),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 cap.get(cv2.CAP_PROP_FPS) or 30,
                                 (int(cap.get(3)), int(cap.get(4))))
    print("Running — press Q to quit")
    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame, counts = predict(model, frame)
        now = time.time()
        frame = draw_panel(frame, counts, 1.0/max(now-prev, 1e-6))
        prev = now
        cv2.namedWindow("Tomato Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Tomato Detection", frame)
        if writer: writer.write(frame)
        if cv2.waitKey(1) in (ord("q"), 27): break
    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {'GPU' if device != 'cpu' else 'CPU'}")
    weights = Path(WEIGHTS)
    if not weights.exists():
        print(f"Weights not found: {weights}")
        print("Run train.py first!"); return
    model = YOLO(str(weights))
    model.to(device)
    try:
        src = int(SOURCE)
        run_stream(model, src)
    except (ValueError, TypeError):
        path = Path(SOURCE)
        if not path.exists():
            print(f"Not found: {path}"); return
        if path.is_dir():       run_folder(model, path)
        elif path.suffix.lower() in IMG_EXTS: run_image(model, path)
        else:                   run_stream(model, str(path))

if __name__ == "__main__":
    main()
