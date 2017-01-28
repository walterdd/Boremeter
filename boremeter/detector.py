import cv2
import numpy as np
import pkg_resources

from bounding_boxes import BoundingBox

DETECTOR_CONFIG = {
    'VJ_cascade_path': pkg_resources.resource_filename('boremeter',
                                                       'cv_haar_cascades/haarcascade_frontalface_default.xml'),
    'cascade_params':  [1.3, 4],
}


def get_faces_vj(img, cascade):
    max_scale, min_neighbors = DETECTOR_CONFIG['cascade_params']
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected = cascade.detectMultiScale(gray, max_scale, min_neighbors)
    bboxs = []
    for face in detected:
        bboxs.append(BoundingBox(*face))
    return bboxs


def filter_faces(bboxs):
    pass
    return bboxs


def check_faces(bboxs):
    pass
    return bboxs


def detect_faces(img, raw_detector='VJ'):

    raw_faces = np.array([])

    if raw_detector == 'VJ':
        detector = cv2.CascadeClassifier(DETECTOR_CONFIG['VJ_cascade_path'])
        raw_faces = get_faces_vj(img, detector)

    filtered_faces = filter_faces(raw_faces)
    checked_faces = check_faces(filtered_faces)

    return checked_faces
