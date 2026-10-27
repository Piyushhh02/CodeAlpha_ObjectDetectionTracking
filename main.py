"""Real-time object detection (YOLOv8) + tracking (SORT) with OpenCV.

Examples:
  python main.py                       # webcam 0
  python main.py --source video.mp4    # video file
  python main.py --source video.mp4 --save output.mp4 --classes person car
"""
import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

from sort import Sort


def color_for(track_id):
    rng = np.random.default_rng(int(track_id) * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def parse_args():
    p = argparse.ArgumentParser(description="Object detection and tracking")
    p.add_argument("--source", default="0", help="webcam index or path to a video file")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO weights (downloaded automatically)")
    p.add_argument("--conf", type=float, default=0.4, help="detection confidence threshold")
    p.add_argument("--classes", nargs="*", help="only track these class names, e.g. person car")
    p.add_argument("--save", help="write annotated video to this path")
    p.add_argument("--no-show", action="store_true", help="do not open a preview window")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    names = model.names
    wanted = None
    if args.classes:
        wanted = {i for i, n in names.items() if n in set(args.classes)}
        if not wanted:
            raise SystemExit(f"None of {args.classes} are valid class names for this model.")

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h)) if args.save else None

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1) Detect
        result = model(frame, conf=args.conf, verbose=False)[0]
        dets = []
        for box, score, cls in zip(result.boxes.xyxy.cpu().numpy(),
                                   result.boxes.conf.cpu().numpy(),
                                   result.boxes.cls.cpu().numpy()):
            if wanted is None or int(cls) in wanted:
                dets.append([*box, score, cls])

        # 2) Track
        tracks = tracker.update(np.array(dets) if dets else np.empty((0, 6)))

        # 3) Draw boxes, labels and IDs
        for x1, y1, x2, y2, tid, cls, score in tracks:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            c = color_for(tid)
            label = f"#{int(tid)} {names[int(cls)]} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), c, -1)
            cv2.putText(frame, label, (x1 + 3, max(th, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        now = time.time()
        fps = 1 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"FPS {fps:.1f}  Tracked {len(tracks)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if writer:
            writer.write(frame)
        if not args.no_show:
            cv2.imshow("Object Detection and Tracking (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
