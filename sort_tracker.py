"""
sort_tracker.py
---------------
精简版 SORT (Simple Online and Realtime Tracking)
只依赖 numpy + scipy,无外部 tracker 库依赖。

参考: Bewley et al., "Simple Online and Realtime Tracking" (2016)

用法:
    tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)
    tracks = tracker.update(detections)  # detections: Nx5 [x1,y1,x2,y2,score]
    # tracks: Mx5 [x1,y1,x2,y2,track_id]
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(bb_test, bb_gt):
    """计算两组 bbox 的 IoU 矩阵. bb_test: Nx4, bb_gt: Mx4."""
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    return wh / (area_test + area_gt - wh + 1e-9)


def convert_bbox_to_z(bbox):
    """[x1,y1,x2,y2] -> [cx, cy, s, r] (s=area, r=aspect ratio)"""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.
    cy = bbox[1] + h / 2.
    s = w * h
    r = w / float(h + 1e-9)
    return np.array([cx, cy, s, r]).reshape(4, 1)


def convert_x_to_bbox(x):
    """[cx, cy, s, r, ...] -> [x1,y1,x2,y2]"""
    w = np.sqrt(max(x[2], 1e-9) * max(x[3], 1e-9))
    h = max(x[2], 1e-9) / (w + 1e-9)
    return np.array([x[0] - w / 2., x[1] - h / 2.,
                     x[0] + w / 2., x[1] + h / 2.]).reshape(1, 4)


class KalmanBoxTracker:
    """单个目标的卡尔曼跟踪器. 状态: [cx, cy, s, r, vx, vy, vs]."""
    count = 0

    def __init__(self, bbox):
        # 简化版KF: 7维状态,4维观测
        self.dim_x = 7
        self.dim_z = 4

        # 状态转移矩阵 F (恒速模型)
        self.F = np.eye(self.dim_x)
        for i in range(3):
            self.F[i, i + 4] = 1.

        # 观测矩阵 H
        self.H = np.zeros((self.dim_z, self.dim_x))
        for i in range(4):
            self.H[i, i] = 1.

        # 协方差
        self.P = np.eye(self.dim_x) * 10.
        self.P[4:, 4:] *= 1000.
        self.Q = np.eye(self.dim_x)
        self.Q[4:, 4:] *= 0.01
        self.R = np.eye(self.dim_z) * 10.

        self.x = np.zeros((self.dim_x, 1))
        self.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.x[:4].flatten()))
        return self.history[-1]

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        z = convert_bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P

    def get_state(self):
        return convert_x_to_bbox(self.x[:4].flatten())


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """匈牙利匹配 detection 和 tracker."""
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.empty((0,), dtype=int))

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # 用 scipy 的匈牙利算法 (最大化 IoU = 最小化 -IoU)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(row_ind, col_ind)))
    else:
        matched_indices = np.empty(shape=(0, 2), dtype=int)

    unmatched_detections = [d for d in range(len(detections))
                            if d not in matched_indices[:, 0]]
    unmatched_trackers = [t for t in range(len(trackers))
                          if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    matches = np.concatenate(matches, axis=0) if matches else np.empty((0, 2), dtype=int)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort:
    """主跟踪器."""

    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        """
        max_age: 目标消失多少帧后删除
        min_hits: 至少匹配多少次才输出 (滤短暂误检)
        iou_threshold: 关联 IoU 阈值
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        dets: Nx5 [x1,y1,x2,y2,score]
        返回: Mx5 [x1,y1,x2,y2,track_id]
        """
        self.frame_count += 1

        # 预测现有 tracker 位置
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # 关联
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets[:, :4] if len(dets) else np.empty((0, 4)),
            trks[:, :4] if len(trks) else np.empty((0, 4)),
            self.iou_threshold
        )

        # 更新匹配上的 tracker
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        # 新建未匹配的 detection
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)

        # 输出 + 清理
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and \
               (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret):
            return np.concatenate(ret)
        return np.empty((0, 5))
