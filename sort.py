"""SORT (Simple Online and Realtime Tracking) - Kalman filter + Hungarian matching.
Reference: Bewley et al., 2016. Self-contained: needs only numpy and scipy.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a, b):
    """Pairwise IoU between boxes a (N,4) and b (M,4) in [x1,y1,x2,y2] format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = a[:, None, :]
    b = b[None, :, :]
    iw = np.maximum(0, np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]))
    ih = np.maximum(0, np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]))
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-9)


def to_z(box):
    """[x1,y1,x2,y2] -> [cx, cy, area, aspect_ratio]."""
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w * h, w / (h + 1e-9)]).reshape(4, 1)


def to_box(x):
    """State vector -> [x1,y1,x2,y2]."""
    w = np.sqrt(max(x[2, 0] * x[3, 0], 1e-9))
    h = x[2, 0] / (w + 1e-9)
    return np.array([x[0, 0] - w / 2, x[1, 0] - h / 2, x[0, 0] + w / 2, x[1, 0] + h / 2])


class KalmanBoxTracker:
    """One tracked object. State: [cx, cy, s, r, vx, vy, vs] with constant-velocity model."""
    count = 0

    def __init__(self, box, cls=-1, score=0.0):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.eye(4, 7)
        self.R = np.diag([1, 1, 10, 10]).astype(float)
        self.P = np.diag([10, 10, 10, 10, 1e4, 1e4, 1e4]).astype(float)
        self.Q = np.diag([1, 1, 1, 1, 0.01, 0.01, 1e-4]).astype(float)
        self.x = np.zeros((7, 1))
        self.x[:4] = to_z(box)
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.cls, self.score = int(cls), float(score)
        self.hits, self.hit_streak, self.time_since_update, self.age = 1, 1, 0, 0

    def predict(self):
        if self.x[6, 0] + self.x[2, 0] <= 0:  # keep area positive
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return to_box(self.x)

    def update(self, box, cls, score):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.cls, self.score = int(cls), float(score)
        y = to_z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    @property
    def box(self):
        return to_box(self.x)


class Sort:
    def __init__(self, max_age=20, min_hits=3, iou_threshold=0.3):
        self.max_age, self.min_hits, self.iou_threshold = max_age, min_hits, iou_threshold
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(self, dets):
        """dets: (N,6) array [x1,y1,x2,y2,score,class]. Returns (K,7): [x1,y1,x2,y2,id,class,score]."""
        self.frame_count += 1
        dets = np.asarray(dets, dtype=float).reshape(-1, 6)

        preds = np.array([t.predict() for t in self.trackers]).reshape(-1, 4)
        valid = ~np.isnan(preds).any(axis=1)
        self.trackers = [t for t, v in zip(self.trackers, valid) if v]
        preds = preds[valid]

        matched, unmatched_dets = [], list(range(len(dets)))
        if len(preds) and len(dets):
            iou = iou_matrix(dets[:, :4], preds)
            rows, cols = linear_sum_assignment(-iou)
            unmatched_dets = [d for d in range(len(dets)) if d not in rows]
            for r, c in zip(rows, cols):
                if iou[r, c] < self.iou_threshold:
                    unmatched_dets.append(r)
                else:
                    matched.append((r, c))

        for r, c in matched:
            self.trackers[c].update(dets[r, :4], dets[r, 5], dets[r, 4])
        for d in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[d, :4], dets[d, 5], dets[d, 4]))

        out = []
        for t in self.trackers:
            if t.time_since_update < 1 and (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                out.append([*t.box, t.id, t.cls, t.score])
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return np.array(out).reshape(-1, 7)
