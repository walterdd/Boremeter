import cv2
import pkg_resources

from .bounding_boxes import BoundingBox

DETECTOR_CONFIG = {
    'VJ_cascade_path': pkg_resources.resource_filename('boremeter',
                                                       'cv_haar_cascades/haarcascade_frontalface_default.xml'),
    'cascade_params':  [1.3, 4],
}


def get_faces_vj(img, cascade):
    max_scale, min_neighbors = DETECTOR_CONFIG['cascade_params']
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected = cascade.detectMultiScale(gray, max_scale, min_neighbors)
    bboxes = [BoundingBox(*face) for face in detected]
    return bboxes


def filter_faces(bboxes):
    raise NotImplementedError()


def check_faces(bboxes):
    raise NotImplementedError()


def detect_faces(img, raw_detector='VJ'):

    if raw_detector == 'VJ':
        detector = cv2.CascadeClassifier(DETECTOR_CONFIG['VJ_cascade_path'])
        raw_faces = get_faces_vj(img, detector)
    else:
        raise RuntimeError('Detection method %s is not supported' % raw_detector)

    return raw_faces
