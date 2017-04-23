import os
import pkg_resources
import gc

import numpy as np
import cv2
import caffe

from .bounding_boxes import bboxes_are_close, have_intersection, convert_bbox, BoundingBox
from .mtcnn_detector import mtcnn_detect

DETECTOR_CONFIG = {
    'VJ_cascade_path': pkg_resources.resource_filename('boremeter',
                                                       'cv_haar_cascades/haarcascade_frontalface_default.xml'),
    'VJ_cascade_params':  [1.3, 4],
    'mtcnn_threshold': [0.6, 0.7, 0.7],
    'mtcnn_factor': 0.709,
    'mtcnn_minsize': 20,
}


def filter_faces(bboxes):
    # TODO: implement function which takes bboxes on a frame and returns only those which seem to contain a face
    raise NotImplementedError()


def check_faces(bboxes):
    # TODO: implement function which checks if there are faces in filtered bboxes
    raise NotImplementedError()


class Tracker:
    def __init__(self, caffe_models_path, timeout=100, detection_method='mtcnn'):
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
        self.detection_method = detection_method
        if detection_method == 'mtcnn':
            self.PNet = caffe.Net(os.path.join(caffe_models_path, "det1.prototxt"),
                                  os.path.join(caffe_models_path, "det1.caffemodel"), caffe.TEST)
            self.RNet = caffe.Net(os.path.join(caffe_models_path, "det2.prototxt"),
                                  os.path.join(caffe_models_path, "det2.caffemodel"), caffe.TEST)
            self.ONet = caffe.Net(os.path.join(caffe_models_path, "det3.prototxt"),
                                  os.path.join(caffe_models_path, "det3.caffemodel"), caffe.TEST)

        def __del__(self):
            if self.detection_method == 'mtcnn':
                del self.PNet
                del self.RNet
                del self.ONet
                gc.collect()


    def detect_faces(self, img, grey_scale=False):
        if self.detection_method == 'VJ':
            detector = cv2.CascadeClassifier(DETECTOR_CONFIG['VJ_cascade_path'])
            max_scale, min_neighbors = DETECTOR_CONFIG['VJ_cascade_params']
            if not grey_scale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected = detector.detectMultiScale(img, max_scale, min_neighbors)
            bboxes = [BoundingBox(*face) for face in detected]
            return bboxes

        elif self.detection_method == 'mtcnn':
            detected, points = mtcnn_detect(img, self.PNet, self.RNet, self.ONet,
                                            DETECTOR_CONFIG['mtcnn_minsize'],
                                            DETECTOR_CONFIG['mtcnn_threshold'],
                                            DETECTOR_CONFIG['mtcnn_factor'],
                                            fastresize=False)
            bboxes = [convert_bbox(face) for face in detected]
            return bboxes

        elif self.detection_method == 'dlib':
            raise NotImplementedError()

        else:
            raise RuntimeError('Detection method %s is not supported' % self.detection_method)

    def read_frame(self, frame):
        self.prev_frame = self.cur_frame
        self.cur_frame = frame
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
                continue
            starting_points = np.float32(self.tracking_points_by_id[cur_id]).reshape(-1, 2)
            ending_points, st, err = cv2.calcOpticalFlowPyrLK(
                cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(self.cur_frame, cv2.COLOR_BGR2GRAY),
                starting_points,
                None,
                **self.lk_params
            )
            ending_points_reversed, st, err = cv2.calcOpticalFlowPyrLK(
                cv2.cvtColor(self.cur_frame, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY),
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
        detected_faces = self.detect_faces(self.cur_frame, grey_scale=False)
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
            points_number = int(self.bboxes_by_id[new_id].w / 2)
            mean = self.bboxes_by_id[new_id].center
            cov = [[self.bboxes_by_id[new_id].w / 3, 0], [0, self.bboxes_by_id[new_id].h / 3]]
            self.tracking_points_by_id[new_id] = np.random.multivariate_normal(mean, cov, points_number)
