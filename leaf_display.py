1# leaf_display.py - Tomato Leaf Disease Detector
# Modes: Webcam | Video | Image | Folder
# Keys: Q=quit  P=pause  S=snapshot  R=record  F=+10s  B=-10s

import cv2
import torch
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from ultralytics import YOLO

# === CHANGE THESE ===
WEIGHTS      = "C:/code/Tomato/runs/tomato_leaf/weights/best.pt"
SOURCE = r"C:\Users\Nadhu\Downloads\stock-footage-diseased-tomato-plant-with-dried-and-spotted-leaves-inside-greenhouse-crop-failure-risk-farming.webm"
SAVE         = False
SNAPSHOT_DIR = Path("C:/code/Tomato/snapshots")
OUTPUT_DIR   = Path("C:/code/Tomato/leaf_outputs")
# ====================

CLASS_NAMES = [
    "Bacterial Spot",     "Early Blight",       "Healthy",
    "Late Blight",        "Leaf Mold",           "Septoria Leaf Spot",
    "Spider Mites",       "Target Spot",         "Mosaic Virus",
    "Yellow Leaf Curl",
]

CLASS_COLORS = {cls: (255, 255, 255) for cls in CLASS_NAMES}

DISEASE_ALERT_COLOR  = (20,  20, 120)
HEALTHY_HEADER_COLOR = (20,  80,  20)

CONF     = 0.40
IOU      = 0.45
IMG_SIZE = 640
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp"}
FONT     = cv2.FONT_HERSHEY_SIMPLEX
BOLD     = cv2.FONT_HERSHEY_DUPLEX


# ══════════════════════════════════════════════════════
#  DRAWING
# ══════════════════════════════════════════════════════

def draw_boxes(frame, results):
    counts = defaultdict(int)
    boxes  = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return frame, counts

    for box in boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        name   = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        color  = CLASS_COLORS.get(name, (255, 255, 255))
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        c, t = 16, 4
        for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (px, py), (px + dx*c, py), color, t)
            cv2.line(frame, (px, py), (px, py + dy*c), color, t)

        label = f" {name}  {conf:.0%} "
        (lw, lh), _ = cv2.getTextSize(label, FONT, 0.52, 1)
        ly = max(y1 - lh - 6, 0)
        lbl_color = (30, 160, 30) if "healthy" in name.lower() else (30, 30, 180)
        cv2.rectangle(frame, (x1, ly), (x1 + lw, ly + lh + 6), lbl_color, -1)
        cv2.putText(frame, label, (x1, ly + lh + 2),
                    FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        counts[name] += 1

    return frame, counts


def draw_panel(frame, counts, fps=0.0, source_label=""):
    h, w  = frame.shape[:2]
    total = sum(counts.values())
    pw    = 230
    ph    = 50 + len(CLASS_NAMES) * 26 + 50
    px    = w - pw - 12
    py    = 12

    ov = frame.copy()
    cv2.rectangle(ov, (px-8, py-8), (px+pw, py+ph), (10, 10, 10), -1)
    cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (px-8, py-8), (px+pw, py+ph), (60, 60, 60), 1)

    cv2.putText(frame, "LEAF DISEASE MONITOR", (px, py+18),
                BOLD, 0.48, (180, 220, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (px-8, py+26), (px+pw, py+26), (50, 50, 50), 1)

    y = py + 46
    for cls in CLASS_NAMES:
        cnt        = counts.get(cls, 0)
        is_healthy = "healthy" in cls.lower()
        dot_color  = (40, 200, 40)   if is_healthy else (255, 255, 255)
        txt_color  = (160, 255, 160) if is_healthy else (220, 220, 220)
        active     = cnt > 0

        cv2.circle(frame, (px+8, y-4), 5,
                   dot_color if active else (50, 50, 50), -1)
        cv2.putText(frame, cls, (px+20, y), FONT, 0.42,
                    txt_color if active else (70, 70, 70), 1, cv2.LINE_AA)
        s = str(cnt)
        (sw, _), _ = cv2.getTextSize(s, BOLD, 0.52, 1)
        cv2.putText(frame, s, (px+pw-sw-6, y), BOLD, 0.52,
                    dot_color if active else (50, 50, 50), 1, cv2.LINE_AA)
        y += 26

    cv2.line(frame, (px-8, y-2), (px+pw, y-2), (50, 50, 50), 1)
    cv2.putText(frame, "TOTAL", (px, y+22),
                BOLD, 0.58, (255, 255, 140), 1, cv2.LINE_AA)
    ts = str(total)
    (tw, _), _ = cv2.getTextSize(ts, BOLD, 0.80, 2)
    cv2.putText(frame, ts, (px+pw-tw-6, y+24),
                BOLD, 0.80, (80, 255, 80), 2, cv2.LINE_AA)

    if fps > 0:
        cv2.putText(frame, f"FPS {fps:.1f}", (px, py+ph-4),
                    FONT, 0.38, (80, 80, 80), 1, cv2.LINE_AA)
    if source_label:
        cv2.putText(frame, source_label, (10, h-10),
                    FONT, 0.38, (70, 70, 70), 1, cv2.LINE_AA)
    return frame


def draw_title_bar(frame_w, counts, elapsed, frame_no=0, total_frames=0):
    diseases = [c for c in counts if "healthy" not in c.lower() and counts[c] > 0]
    bg = DISEASE_ALERT_COLOR if diseases else HEALTHY_HEADER_COLOR
    bar = np.zeros((36, frame_w, 3), dtype=np.uint8)
    bar[:] = bg

    left = f"TOMATO LEAF MONITOR   T: {elapsed:.0f}s"
    if total_frames > 0:
        left += f"   Frame: {frame_no}/{total_frames}"
    cv2.putText(bar, left, (10, 24), FONT, 0.52, (0, 220, 150), 1)

    hint = "Q=quit  P=pause  S=snapshot  R=record"
    (hw, _), _ = cv2.getTextSize(hint, FONT, 0.42, 1)
    cv2.putText(bar, hint, (frame_w - hw - 10, 24), FONT, 0.42, (120, 120, 120), 1)

    if diseases:
        alert = "DISEASE: " + " | ".join(diseases)[:55]
        (aw, _), _ = cv2.getTextSize(alert, FONT, 0.48, 1)
        cv2.putText(bar, alert, (frame_w//2 - aw//2, 24),
                    FONT, 0.48, (80, 80, 255), 1)
    return bar


def stack_frame(frame, counts, elapsed, frame_no=0, total_frames=0):
    bar = draw_title_bar(frame.shape[1], counts, elapsed, frame_no, total_frames)
    return np.vstack([bar, frame])


def predict(model, frame):
    results = model.predict(frame, conf=CONF, iou=IOU,
                            imgsz=IMG_SIZE, verbose=False)
    return draw_boxes(frame, results)


# ══════════════════════════════════════════════════════
#  CAMERA UTILITIES
# ══════════════════════════════════════════════════════

def find_cameras(max_check=6):
    cams = []
    print("Scanning cameras...")
    for i in range(max_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cams.append({"index": i, "w": w, "h": h})
                print(f"  Camera {i}: {w}x{h}")
        cap.release()
    return cams


def select_camera():
    cams = find_cameras()
    if not cams:
        print("No cameras found!")
        return None
    if len(cams) == 1:
        return cams[0]["index"]
    print("\n  Choose camera:")
    for c in cams:
        print(f"  [{c['index']}] Camera {c['index']}  {c['w']}x{c['h']}")
    try:
        return int(input("  Enter number: ").strip())
    except Exception:
        return cams[0]["index"]


# ══════════════════════════════════════════════════════
#  RUNNERS
# ══════════════════════════════════════════════════════

def run_image(model, img_path):
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"Cannot read: {img_path}")
        return
    frame, counts = predict(model, frame)
    frame  = draw_panel(frame, counts, source_label=img_path.name)
    output = stack_frame(frame, counts, 0)

    print(f"\n{img_path.name}")
    for c in CLASS_NAMES:
        if counts.get(c, 0) > 0:
            print(f"  {c:<25}: {counts[c]}")
    print(f"  {'TOTAL':<25}: {sum(counts.values())}")

    if SAVE:
        OUTPUT_DIR.mkdir(exist_ok=True)
        cv2.imwrite(str(OUTPUT_DIR / img_path.name), output)
        print(f"  Saved -> {OUTPUT_DIR / img_path.name}")

    cv2.namedWindow("Leaf Disease Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Leaf Disease Monitor", 1100, 700)
    cv2.imshow("Leaf Disease Monitor", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_folder(model, folder):
    imgs = sorted([f for f in folder.iterdir()
                   if f.suffix.lower() in IMG_EXTS])
    if not imgs:
        print(f"No images in {folder}")
        return
    grand = defaultdict(int)
    print(f"Processing {len(imgs)} images — press Q to stop")

    for p in imgs:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        frame, counts = predict(model, frame)
        frame  = draw_panel(frame, counts, source_label=p.name)
        output = stack_frame(frame, counts, 0)
        for c, n in counts.items():
            grand[c] += n
        if SAVE:
            OUTPUT_DIR.mkdir(exist_ok=True)
            cv2.imwrite(str(OUTPUT_DIR / p.name), output)
        cv2.namedWindow("Leaf Disease Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Leaf Disease Monitor", 1100, 700)
        cv2.imshow("Leaf Disease Monitor", output)
        if cv2.waitKey(600) in (ord("q"), 27):
            break

    cv2.destroyAllWindows()
    print("\nFolder Total:")
    for c in CLASS_NAMES:
        if grand.get(c, 0) > 0:
            print(f"  {c:<25}: {grand[c]}")
    print(f"  {'TOTAL':<25}: {sum(grand.values())}")


def run_video(model, source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open: {source}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_cap      = cap.get(cv2.CAP_PROP_FPS) or 30
    src_name     = Path(str(source)).name
    writer       = None

    if SAVE:
        OUTPUT_DIR.mkdir(exist_ok=True)
        nm = Path(str(source)).stem + "_leaf_det.mp4"
        w  = int(cap.get(3))
        h  = int(cap.get(4))
        writer = cv2.VideoWriter(str(OUTPUT_DIR / nm),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_cap, (w, h + 36))

    print(f"Playing: {src_name}  ({total_frames} frames)")
    print("Q=quit  P=pause  S=snapshot  F=+10s  B=-10s")

    paused   = False
    prev     = time.time()
    start    = time.time()
    frame_no = 0
    smooth   = fps_cap
    last_out = None

    cv2.namedWindow("Leaf Disease Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Leaf Disease Monitor", 1280, 760)

    while True:
        key = cv2.waitKey(1 if not paused else 30) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("p"):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord("s") and last_out is not None:
            SNAPSHOT_DIR.mkdir(exist_ok=True)
            p = str(SNAPSHOT_DIR / f"snap_{datetime.now().strftime('%H%M%S')}.jpg")
            cv2.imwrite(p, last_out)
            print(f"Snapshot -> {p}")
        elif key == ord("f"):
            fn = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    min(fn + int(fps_cap * 10), total_frames - 1))
        elif key == ord("b"):
            fn = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    max(fn - int(fps_cap * 10), 0))

        if paused:
            if last_out is not None:
                cv2.imshow("Leaf Disease Monitor", last_out)
            continue

        ok, frame = cap.read()
        if not ok:
            print("Video ended.")
            break

        frame_no += 1
        frame, counts = predict(model, frame)

        now    = time.time()
        smooth = 0.85 * smooth + 0.15 / max(now - prev, 0.001)
        prev   = now

        frame  = draw_panel(frame, counts, smooth, source_label=src_name)
        output = stack_frame(frame, counts,
                             time.time() - start, frame_no, total_frames)
        last_out = output

        if writer:
            writer.write(output)
        cv2.imshow("Leaf Disease Monitor", output)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done.")


def run_webcam(model, cam_index=None, width=1280, height=720):
    if cam_index is None:
        cam_index = select_camera()
    if cam_index is None:
        print("No camera available.")
        return

    print(f"\nOpening Camera {cam_index}...")
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Cannot open camera {cam_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fw = int(cap.get(3))
    fh = int(cap.get(4))
    print(f"Resolution: {fw}x{fh}")
    print("Q=quit  P=pause  S=snapshot  R=record")

    paused    = False
    recording = False
    writer    = None
    shot_n    = 0
    last_out  = None
    prev      = time.time()
    start     = time.time()
    smooth    = 30.0

    OUTPUT_DIR.mkdir(exist_ok=True)
    cv2.namedWindow("Leaf Disease Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Leaf Disease Monitor", 1280, 760)

    while True:
        key = cv2.waitKey(1 if not paused else 30) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("p"):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord("s") and last_out is not None:
            shot_n += 1
            SNAPSHOT_DIR.mkdir(exist_ok=True)
            p = str(SNAPSHOT_DIR / f"snap_{shot_n:04d}_{datetime.now().strftime('%H%M%S')}.jpg")
            cv2.imwrite(p, last_out)
            print(f"Snapshot -> {p}")
        elif key == ord("r"):
            if not recording:
                rec_path = OUTPUT_DIR / f"webcam_{int(time.time())}.mp4"
                writer   = cv2.VideoWriter(
                    str(rec_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    20, (fw, fh + 36))
                recording = True
                print(f"Recording -> {rec_path}")
            else:
                if writer:
                    writer.release()
                    writer = None
                recording = False
                print("Recording stopped.")

        if paused:
            if last_out is not None:
                cv2.imshow("Leaf Disease Monitor", last_out)
            continue

        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed.")
            break

        frame, counts = predict(model, frame)

        now    = time.time()
        smooth = 0.85 * smooth + 0.15 / max(now - prev, 0.001)
        prev   = now

        frame  = draw_panel(frame, counts, smooth)
        output = stack_frame(frame, counts, time.time() - start)

        # Webcam badge
        badge = f"  WEBCAM {cam_index}  "
        (bw, bh), _ = cv2.getTextSize(badge, BOLD, 0.50, 1)
        cv2.rectangle(output, (8, 44), (8+bw+4, 44+bh+8), (0, 160, 0), -1)
        cv2.putText(output, badge, (10, 44+bh+2),
                    BOLD, 0.50, (0, 0, 0), 1, cv2.LINE_AA)

        if recording:
            cv2.circle(output, (output.shape[1]-20, 50), 8, (0, 0, 220), -1)
            cv2.putText(output, "REC", (output.shape[1]-55, 56),
                        FONT, 0.40, (0, 0, 220), 1)

        last_out = output
        if recording and writer:
            writer.write(output)
        cv2.imshow("Leaf Disease Monitor", output)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


# ══════════════════════════════════════════════════════
#  STARTUP MENU
# ══════════════════════════════════════════════════════

def startup_menu():
    print("\n" + "=" * 46)
    print("     TOMATO LEAF DISEASE DETECTOR")
    print("=" * 46)
    print("  [1]  Webcam  (live camera)")
    print("  [2]  Video File")
    print("  [3]  Image File")
    print("  [4]  Image Folder")
    print("=" * 46)
    choice = input("  Select mode (1/2/3/4): ").strip()
    if choice == "1":
        return "webcam", None
    elif choice == "2":
        return "video",  input("  Video path : ").strip().strip('"')
    elif choice == "3":
        return "image",  input("  Image path : ").strip().strip('"')
    elif choice == "4":
        return "folder", input("  Folder path: ").strip().strip('"')
    else:
        print("  Invalid choice.")
        return None, None


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

def main():
    global SAVE, CONF   # declared first before any use

    parser = argparse.ArgumentParser(description="Tomato Leaf Disease Detector")
    parser.add_argument("--source",  default=None)
    parser.add_argument("--webcam",  action="store_true")
    parser.add_argument("--cam",     type=int, default=None)
    parser.add_argument("--width",   type=int, default=1280)
    parser.add_argument("--height",  type=int, default=720)
    parser.add_argument("--save",    action="store_true")
    parser.add_argument("--conf",    type=float, default=CONF)
    args = parser.parse_args()

    if args.save:
        SAVE = True
    CONF = args.conf

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {'GPU (CUDA)' if device != 'cpu' else 'CPU'}")

    weights = Path(WEIGHTS)
    if not weights.exists():
        print(f"Weights not found: {weights}")
        print("Train a leaf model first!")
        return

    print(f"Loading: {weights.name}")
    model = YOLO(str(weights))
    model.to(device)

    # No CLI args → show startup menu
    if not args.webcam and args.cam is None and args.source is None:
        mode, src = startup_menu()
        if mode is None:
            return
        elif mode == "webcam":
            run_webcam(model, width=args.width, height=args.height)
        elif mode == "video":
            p = Path(src)
            if not p.exists():
                print(f"Not found: {p}"); return
            run_video(model, str(p))
        elif mode == "image":
            p = Path(src)
            if not p.exists():
                print(f"Not found: {p}"); return
            run_image(model, p)
        elif mode == "folder":
            p = Path(src)
            if not p.exists():
                print(f"Not found: {p}"); return
            run_folder(model, p)
        return

    # CLI args given
    if args.webcam or args.cam is not None:
        run_webcam(model, cam_index=args.cam,
                   width=args.width, height=args.height)
        return

    src = args.source if args.source else SOURCE
    try:
        run_webcam(model, cam_index=int(src),
                   width=args.width, height=args.height)
        return
    except (ValueError, TypeError):
        pass

    path = Path(src)
    if not path.exists():
        print(f"Not found: {path}")
        return

    if path.is_dir():
        run_folder(model, path)
    elif path.suffix.lower() in IMG_EXTS:
        run_image(model, path)
    else:
        run_video(model, str(path))


if __name__ == "__main__":
    main()