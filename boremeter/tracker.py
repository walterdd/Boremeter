import cv2
from detector import detect_faces
from bounding_boxes import bboxes_are_close, have_intersection, BoundingBox
import numpy as np


class Tracker:
    def __init__(self, timeout=400):
        self.timeout = timeout
        self.cur_frame_num = -1
        self.prev_frame = None
        self.cur_frame = None
        self.tracking_points_by_id = {}
        self.bboxes_by_id = {}
        self.last_detection_frame_by_id = {}
        self.max_id = 0
        self.lk_params = dict(winSize=(3, 3),
                              maxLevel=0,
                              criteria=(cv2.TERM_CRITERIA_EPS |
                                        cv2.TERM_CRITERIA_COUNT,
                                        10,
                                        0.03)
                              )

    def read_frame(self, frame):
        self.prev_frame = self.cur_frame
        self.cur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.cur_frame_num += 1

    def track_faces(self):
        if self.prev_frame is None:
            return
        lost_faces = []
        for cur_id, cur_bbox in self.bboxes_by_id.iteritems():

            if self.cur_frame_num - self.last_detection_frame_by_id[cur_id] > self.timeout:
                lost_faces.append(cur_id)
                continue
            if len(self.tracking_points_by_id[cur_id]) == 0:
                points_number = self.bboxes_by_id[cur_id].w / 2
                mean = self.bboxes_by_id[cur_id].center
                cov = [[self.bboxes_by_id[cur_id].w / 3, 0], [0, self.bboxes_by_id[cur_id].h / 3]]
                self.tracking_points_by_id[cur_id] = np.random.multivariate_normal(mean, cov, points_number)
                continue
            starting_points = np.float32(self.tracking_points_by_id[cur_id]).reshape(-1, 2)
            ending_points, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_frame,
                self.cur_frame,
                starting_points,
                None,
                **self.lk_params
            )
            ending_points_reversed, st, err = cv2.calcOpticalFlowPyrLK(
                self.cur_frame,
                self.prev_frame,
                ending_points,
                None,
                **self.lk_params
            )
            new_points = []
            stable_points = abs(starting_points-ending_points_reversed).max(-1) < cur_bbox.w / 10
            for tr, (x, y), good_flag in zip(self.tracking_points_by_id[cur_id],
                                             ending_points.reshape(-1, 2),
                                             stable_points):
                if not good_flag:
                    continue
                new_points.append((x, y))
            self.tracking_points_by_id[cur_id] = new_points
            points_number = len(self.tracking_points_by_id[cur_id])
            if points_number > 0:
                center_x, center_y = np.mean(self.tracking_points_by_id[cur_id], axis=0) if points_number > 1 \
                    else self.tracking_points_by_id[cur_id][0]

                self.bboxes_by_id[cur_id] = BoundingBox(
                    center_x - cur_bbox.w / 2,
                    center_y - cur_bbox.h / 2,
                    cur_bbox.w,
                    cur_bbox.h
                )
            elif len(ending_points) > 1:
                approx_center_x, approx_center_y = np.mean(ending_points, axis=0)
                cur_center_x, cur_center_y = cur_bbox.center
                self.bboxes_by_id[cur_id] = BoundingBox(
                    (approx_center_x + cur_center_x) / 2 - cur_bbox.w / 2,
                    (approx_center_y + cur_center_y) / 2 - cur_bbox.h / 2,
                    cur_bbox.w,
                    cur_bbox.h
                )
        for lost_id in lost_faces:
            del self.bboxes_by_id[lost_id]
            del self.tracking_points_by_id[lost_id]
            del self.last_detection_frame_by_id[lost_id]

    def match_faces(self):
        detected_faces = detect_faces(self.cur_frame, grey_scale=True)
        new_faces = {}
        for bbox in detected_faces:
            self.max_id += 1
            new_faces[self.max_id] = bbox
        ids_to_get_points = []
        for new_id, new_bbox in new_faces.iteritems():
            face_untracked = True
            for old_id, old_bbox in self.bboxes_by_id.iteritems():
                if bboxes_are_close(new_bbox, old_bbox):
                    self.bboxes_by_id[old_id] = new_bbox.resize(0.8)
                    self.last_detection_frame_by_id[old_id] = self.cur_frame_num
                    face_untracked = False
                    ids_to_get_points.append(old_id)
                    break
                if have_intersection(old_bbox, new_bbox):
                    face_untracked = False
                    break
            if face_untracked:
                self.bboxes_by_id[new_id] = new_bbox.resize(0.8)
                self.last_detection_frame_by_id[new_id] = self.cur_frame_num
                ids_to_get_points.append(new_id)
        for new_id in ids_to_get_points:
            points_number = self.bboxes_by_id[new_id].w / 2
            mean = self.bboxes_by_id[new_id].center
            cov = [[self.bboxes_by_id[new_id].w / 3, 0], [0, self.bboxes_by_id[new_id].h / 3]]
            self.tracking_points_by_id[new_id] = np.random.multivariate_normal(mean, cov, points_number)
