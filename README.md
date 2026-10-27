# CodeAlpha_ObjectDetectionTracking

Real-time object detection and tracking (Task 4).

## Pipeline
1. **Video input** - webcam or video file through OpenCV.
2. **Detection** - pre-trained YOLOv8 (COCO, 80 classes) on every frame.
3. **Tracking** - SORT, implemented in `sort.py`: a Kalman filter predicts each object's next position and the Hungarian algorithm matches predictions to new detections by IoU.
4. **Display** - bounding boxes with class label, confidence and a persistent tracking ID, plus live FPS.

## Run
```bash
pip install -r requirements.txt
python main.py                                   # webcam
python main.py --source video.mp4 --save out.mp4 # file in, annotated file out
python main.py --classes person car --conf 0.5   # filter classes
```
Press `q` to quit. `yolov8n.pt` downloads on first run; use `--model yolov8s.pt` for higher accuracy.

## Files
- `main.py` - capture, detection, drawing
- `sort.py` - SORT tracker (numpy + scipy only)
