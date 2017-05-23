import os

import pandas as pd
import cv2
from tqdm import tqdm

from .tracker import Tracker, Detector

def crop_faces(img, frame_num, bboxes, tmp_dir):
    for person_id, bbox in bboxes.iteritems():
        bbox = bbox.resize(scale=3)
        crop_file_path = os.path.join(tmp_dir, 'person%dframe%d.jpg' % (person_id, frame_num))
        crop_img = img[bbox.top:bbox.bottom, bbox.left:bbox.right]
        cv2.imwrite(crop_file_path, crop_img)


def extract_faces(video_file_path, frames_limit, tmp_dir, detection_step, caffe_models_path):
    faces_df = pd.DataFrame(columns=['frame', 'person_id', 'x', 'y', 'w', 'h', 'points'])

    input_video = cv2.VideoCapture(video_file_path)
    detector = Detector(caffe_models_path=caffe_models_path, detection_method='mtcnn')
    tracker = Tracker(detector)
    for _ in tqdm(range(frames_limit)):
        has_more_frames, cur_frame = input_video.read()
        if not has_more_frames:
            break

        tracker.read_frame(cur_frame)
        tracker.track_faces()
        if tracker.cur_frame_num % detection_step == 0:
            tracker.match_faces()

        new_bboxes_by_id = tracker.bboxes_by_id
        new_landmarks_by_id = tracker.landmarks_by_id
        crop_faces(tracker.cur_frame, tracker.cur_frame_num, new_bboxes_by_id, tmp_dir)

        if tracker.cur_frame_num % detection_step == 0:
            for person_id in new_bboxes_by_id:
                face = new_bboxes_by_id[person_id]
                landmarks = new_landmarks_by_id[person_id]
                faces_df = faces_df.append({'frame': tracker.cur_frame_num,
                                            'person_id': person_id,
                                            'x': face.x,
                                            'y': face.y,
                                            'w': face.w,
                                            'h': face.h,
                                            'points': ','.join([str(l) for l in landmarks])},
                                           ignore_index=True,)
        else:
            for person_id in new_bboxes_by_id:
                face = new_bboxes_by_id[person_id]
                faces_df = faces_df.append({'frame': tracker.cur_frame_num,
                                            'person_id': person_id,
                                            'x': face.x,
                                            'y': face.y,
                                            'w': face.w,
                                            'h': face.h,
                                            'points': None},
                                           ignore_index=True,)
    del tracker
    return faces_df, cur_frame.shape
