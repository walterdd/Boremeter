import cv2
import numpy as np

global_config = {
    'VJ_cascade_path' : '../Project/cv_haar_cascades/haarcascade_frontalface_default.xml',
    'cascade_params'  : [1.15, 3]
}

def get_faces_VJ(img, cascade):

    max_scale, min_neighbors = global_config['cascade_params']
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cascade.detectMultiScale(gray, max_scale, min_neighbors)


def filter_faces(bbs):

    return bbs


def check_faces(bbs):

    return bbs


def detect_faces(img, raw_detector='VJ'):

    raw_faces = np.array([])

    if raw_detector == 'VJ':
        detector = cv2.CascadeClassifier(global_config['VJ_cascade_path'])
        raw_faces = get_faces_VJ(img, detector)

    filtered_faces = filter_faces(raw_faces)
    checked_faces = check_faces(filtered_faces)

    return checked_faces


def detect_faces_on_video(video_file_path, detection_step=1, frames_limit=200):

    input_video = cv2.VideoCapture(video_file_path)

    cur_frame = 0    
    frames = {}

    ret = True

    while cur_frame < frames_limit and ret:
        ret, frame = input_video.read() 

        frames[cur_frame] = [0,1]
        frames[cur_frame][0] = frame

        if cur_frame % detection_step == 0:
            frames[cur_frame][1] = detect_faces(frame)
        else:
            frames[cur_frame][1] = np.array([])

        cur_frame += 1
    return frames
