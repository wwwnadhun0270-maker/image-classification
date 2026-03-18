# display.py - Tomato Detector (Damaged / Old / Ripe / Unripe)
# Features: Webcam + Video + Image + Folder + Startup Menu

import cv2
import torch
import time
import argparse
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# === CHANGE THESE ===
WEIGHTS = "C:/code/Tomato/runs/tomato_v3/weights/best.pt"
SOURCE  = "C:/Users/Nadhu/Downloads/4688-179650983_medium.mp4"
SAVE    = False
# ====================

CLASS_NAMES  = ["Damaged", "Old", "Ripe", "Unripe"]
CONF         = 0.25
IOU          = 0.40
IMG_SIZE     = 640
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp"}
OUTPUT_DIR   = Path("C:/code/Tomato/tomato_outputs")

CLASS_COLORS = {
    "Damaged": (0,   40,  210),
    "Old":     (0,  140,  255),
    "Ripe":    (30, 200,   30),
    "Unripe":  (20, 180,  180),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX
BOLD = cv2.FONT_HERSHEY_DUPLEX


# ══════════════════════════════════════════════════════
#  WEBCAM UTILITIES
# ══════════════════════════════════════════════════════

def find_available_cameras(max_check=6):
    """Scan camera indices 0-max_check and return available ones."""
    available = []
    print("Scanning for cameras...")
    for i in range(max_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append({"index": i, "width": w, "height": h})
                print(f"  Camera {i}: {w}x{h}  OK")
            cap.release()
    return available


def select_camera_interactive():
    """Let user pick a camera from console. Returns camera index."""
    cams = find_available_cameras()
    if not cams:
        print("No cameras found!")
        return None
    if len(cams) == 1:
        print(f"Only one camera found — using Camera {cams[0]['index']}")
        return cams[0]["index"]
    print("\nAvailable cameras:")
    for c in cams:
        print(f"  [{c['index']}] Camera {c['index']}  ({c['width']}x{c['height']})")
    while True:
        try:
            choice = int(input("\nEnter camera number: ").strip())
            if any(c["index"] == choice for c in cams):
                return choice
            print("Invalid choice, try again.")
        except (ValueError, KeyboardInterrupt):
            return cams[0]["index"]


def set_camera_resolution(cap, width=1280, height=720):
    """Try to set webcam resolution."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam resolution: {actual_w}x{actual_h}")
    return actual_w, actual_h


# ══════════════════════════════════════════════════════
#  DRAWING HELPERS
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
        color  = CLASS_COLORS.get(name, (255, 200, 0))
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Corner accents
        c, t = 14, 4
        for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (px, py), (px + dx*c, py), color, t)
            cv2.line(frame, (px, py), (px, py + dy*c), color, t)

        # Label
        label = f" {name} {conf:.0%} "
        (lw, lh), _ = cv2.getTextSize(label, FONT, 0.52, 1)
        ly = max(y1 - lh - 6, 0)
        cv2.rectangle(frame, (x1, ly), (x1 + lw, ly + lh + 6), color, -1)
        cv2.putText(frame, label, (x1, ly + lh + 2),
                    FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        counts[name] += 1
    return frame, counts


def draw_panel(frame, counts, fps=0.0, source_label=""):
    """Side panel: per-class counts + total + fps."""
    h, w  = frame.shape[:2]
    total = sum(counts.values())
    pw    = 210
    ph    = 50 + len(CLASS_NAMES) * 30 + 50
    px    = w - pw - 12
    py    = 12

    ov = frame.copy()
    cv2.rectangle(ov, (px-8, py-8), (px+pw, py+ph), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (px-8, py-8), (px+pw, py+ph), (70, 70, 70), 1)

    cv2.putText(frame, "TOMATO COUNTER", (px, py+18),
                BOLD, 0.52, (200, 200, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (px-8, py+26), (px+pw, py+26), (60, 60, 60), 1)

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

    if source_label:
        cv2.putText(frame, source_label, (10, h - 10),
                    FONT, 0.40, (80, 80, 80), 1, cv2.LINE_AA)
    return frame


def draw_webcam_overlay(frame, cam_index, paused=False, recording=False):
    """Webcam badge + REC dot + hotkey hints."""
    h, w = frame.shape[:2]

    badge_color = (0, 200, 0) if not paused else (0, 140, 255)
    badge_text  = f"  WEBCAM {cam_index}  " if not paused else "  PAUSED  "
    (bw, bh), _ = cv2.getTextSize(badge_text, BOLD, 0.52, 1)
    cv2.rectangle(frame, (8, 8), (8 + bw + 4, 8 + bh + 8), badge_color, -1)
    cv2.putText(frame, badge_text, (10, 8 + bh + 2),
                BOLD, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

    if recording:
        cv2.circle(frame, (w - 20, 20), 8, (0, 0, 220), -1)
        cv2.putText(frame, "REC", (w - 50, 26),
                    FONT, 0.40, (0, 0, 220), 1, cv2.LINE_AA)

    hints = "[Q] Quit  [P] Pause  [S] Screenshot  [R] Record"
    cv2.putText(frame, hints, (10, h - 10),
                FONT, 0.36, (70, 70, 70), 1, cv2.LINE_AA)
    return frame


def predict(model, frame):
    results = model.predict(frame, conf=CONF, iou=IOU,
                            imgsz=IMG_SIZE, verbose=False)
    return draw_boxes(frame, results)


# ══════════════════════════════════════════════════════
#  RUNNERS
# ══════════════════════════════════════════════════════

def run_image(model, img_path):
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"Cannot read: {img_path}"); return
    frame, counts = predict(model, frame)
    frame = draw_panel(frame, counts, source_label=img_path.name)
    print(f"\n{img_path.name}")
    for c in CLASS_NAMES:
        print(f"  {c:<10}: {counts.get(c, 0)}")
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
    imgs = sorted([f for f in folder.iterdir()
                   if f.suffix.lower() in IMG_EXTS])
    if not imgs:
        print(f"No images in {folder}"); return
    grand = defaultdict(int)
    print(f"Processing {len(imgs)} images — press Q to stop")
    for p in imgs:
        frame = cv2.imread(str(p))
        if frame is None: continue
        frame, counts = predict(model, frame)
        frame = draw_panel(frame, counts, source_label=p.name)
        for c, n in counts.items(): grand[c] += n
        if SAVE:
            OUTPUT_DIR.mkdir(exist_ok=True)
            cv2.imwrite(str(OUTPUT_DIR / p.name), frame)
        cv2.namedWindow("Tomato Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tomato Detection", 960, 640)
        cv2.imshow("Tomato Detection", frame)
        if cv2.waitKey(600) in (ord("q"), 27): break
    cv2.destroyAllWindows()
    print("\nFolder Total:")
    for c in CLASS_NAMES:
        print(f"  {c:<10}: {grand.get(c, 0)}")
    print(f"  {'TOTAL':<10}: {sum(grand.values())}")


def run_video(model, source):
    """Video file with detection + pause support."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open video: {source}"); return

    writer = None
    if SAVE:
        OUTPUT_DIR.mkdir(exist_ok=True)
        nm = Path(str(source)).stem + "_det.mp4"
        writer = cv2.VideoWriter(
            str(OUTPUT_DIR / nm),
            cv2.VideoWriter_fourcc(*"mp4v"),
            cap.get(cv2.CAP_PROP_FPS) or 30,
            (int(cap.get(3)), int(cap.get(4)))
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_name     = Path(str(source)).name
    print(f"Playing: {src_name}  ({total_frames} frames)")
    print("Press Q/ESC to quit  |  P to pause/resume")

    paused   = False
    prev     = time.time()
    frame_no = 0

    cv2.namedWindow("Tomato Detection — Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tomato Detection — Video", 1280, 720)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27): break
        if key == ord("p"):
            paused = not paused
            print("Paused" if paused else "Resumed")

        if paused:
            continue

        ok, frame = cap.read()
        if not ok:
            print("Video ended.")
            break

        frame_no += 1
        frame, counts = predict(model, frame)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now

        progress = f"{src_name}  [{frame_no}/{total_frames}]"
        frame = draw_panel(frame, counts, fps, source_label=progress)

        # VIDEO badge
        badge = "  VIDEO  "
        (bw, bh), _ = cv2.getTextSize(badge, BOLD, 0.52, 1)
        cv2.rectangle(frame, (8, 8), (8 + bw + 4, 8 + bh + 8), (200, 130, 0), -1)
        cv2.putText(frame, badge, (10, 8 + bh + 2),
                    BOLD, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

        h = frame.shape[0]
        cv2.putText(frame, "[Q] Quit  [P] Pause/Resume",
                    (10, h - 10), FONT, 0.36, (70, 70, 70), 1, cv2.LINE_AA)

        if writer:
            writer.write(frame)

        cv2.imshow("Tomato Detection — Video", frame)

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print("Video closed.")


def run_webcam(model, cam_index=None, width=1280, height=720):
    """
    Webcam mode with:
      P  — Pause / Resume
      S  — Screenshot
      R  — Start / Stop recording
      Q  — Quit
    """
    if cam_index is None:
        cam_index = select_camera_interactive()
    if cam_index is None:
        print("No camera available."); return

    print(f"\nOpening Camera {cam_index}...")
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Cannot open camera {cam_index}"); return

    set_camera_resolution(cap, width, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    paused   = False
    recording= False
    writer   = None
    shot_n   = 0
    last_frame = None

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Webcam running!")
    print("  [Q/ESC] Quit  [P] Pause  [S] Screenshot  [R] Record")

    cv2.namedWindow("Tomato Detection — Webcam", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tomato Detection — Webcam", 1280, 720)

    prev = time.time()

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        if key == ord("p"):
            paused = not paused
            print("Paused" if paused else "Resumed")

        if key == ord("s") and last_frame is not None:
            shot_n += 1
            fname = OUTPUT_DIR / f"webcam_shot_{shot_n:04d}.jpg"
            cv2.imwrite(str(fname), last_frame)
            print(f"Screenshot saved: {fname}")

        if key == ord("r"):
            if not recording:
                rec_path = OUTPUT_DIR / f"webcam_rec_{int(time.time())}.mp4"
                writer = cv2.VideoWriter(
                    str(rec_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    20, (frame_w, frame_h)
                )
                recording = True
                print(f"Recording started: {rec_path}")
            else:
                if writer: writer.release(); writer = None
                recording = False
                print("Recording stopped.")

        if paused:
            if last_frame is not None:
                disp = draw_webcam_overlay(last_frame.copy(), cam_index,
                                           paused=True, recording=recording)
                cv2.imshow("Tomato Detection — Webcam", disp)
            continue

        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed.")
            break

        frame, counts = predict(model, frame)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now

        frame = draw_panel(frame, counts, fps)
        frame = draw_webcam_overlay(frame, cam_index,
                                    paused=False, recording=recording)
        last_frame = frame.copy()

        if recording and writer:
            writer.write(frame)

        cv2.imshow("Tomato Detection — Webcam", frame)

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


# ══════════════════════════════════════════════════════
#  STARTUP MENU
# ══════════════════════════════════════════════════════

def startup_menu():
    """Console menu shown when no CLI args given."""
    print("\n" + "=" * 44)
    print("        TOMATO DETECTOR")
    print("=" * 44)
    print("  [1]  Webcam  (live camera)")
    print("  [2]  Video File")
    print("  [3]  Image File")
    print("  [4]  Image Folder")
    print("=" * 44)

    choice = input("Select mode (1/2/3/4): ").strip()

    if choice == "1":
        return "webcam", None

    elif choice == "2":
        path = input("Video file path: ").strip().strip('"')
        return "video", path

    elif choice == "3":
        path = input("Image file path: ").strip().strip('"')
        return "image", path

    elif choice == "4":
        path = input("Folder path: ").strip().strip('"')
        return "folder", path

    else:
        print("Invalid choice.")
        return None, None


# ══════════════════════════════════════════════════════
#  ARG PARSER
# ══════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Tomato Detector")
    p.add_argument("--source",  default=None,
                   help="Image / video / folder path.")
    p.add_argument("--webcam",  action="store_true",
                   help="Force webcam mode")
    p.add_argument("--cam",     type=int, default=None,
                   help="Camera index  e.g. --cam 0")
    p.add_argument("--width",   type=int, default=1280)
    p.add_argument("--height",  type=int, default=720)
    p.add_argument("--save",    action="store_true",
                   help="Save output files")
    return p.parse_args()


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

def main():
    args = parse_args()

    global SAVE
    if args.save:
        SAVE = True

    # Load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {'GPU' if device != 'cpu' else 'CPU'}")

    weights = Path(WEIGHTS)
    if not weights.exists():
        print(f"Weights not found: {weights}")
        print("Run train.py first!")
        return

    model = YOLO(str(weights))
    model.to(device)

    # ── No CLI args → show startup menu ──────────────
    if not args.webcam and args.cam is None and args.source is None:
        mode, src_path = startup_menu()

        if mode is None:
            return
        elif mode == "webcam":
            run_webcam(model, cam_index=None,
                       width=args.width, height=args.height)
        elif mode == "video":
            p = Path(src_path)
            if not p.exists(): print(f"Not found: {p}"); return
            run_video(model, str(p))
        elif mode == "image":
            p = Path(src_path)
            if not p.exists(): print(f"Not found: {p}"); return
            run_image(model, p)
        elif mode == "folder":
            p = Path(src_path)
            if not p.exists(): print(f"Not found: {p}"); return
            run_folder(model, p)
        return

    # ── CLI args given ────────────────────────────────
    if args.webcam or args.cam is not None:
        run_webcam(model, cam_index=args.cam,
                   width=args.width, height=args.height)
        return

    src = args.source if args.source else SOURCE

    try:
        cam_idx = int(src)
        run_webcam(model, cam_index=cam_idx,
                   width=args.width, height=args.height)
        return
    except (ValueError, TypeError):
        pass

    path = Path(src)
    if not path.exists():
        print(f"Not found: {path}"); return

    if path.is_dir():
        run_folder(model, path)
    elif path.suffix.lower() in IMG_EXTS:
        run_image(model, path)
    else:
        run_video(model, str(path))


if __name__ == "__main__":
    main()