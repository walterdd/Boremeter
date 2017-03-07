import os
import csv

import pandas as pd
import cv2
from tqdm import tqdm

from .detector import detect_faces
from .bounding_boxes import have_intersection, bboxes_are_close
from .tracker import Tracker


def crop_faces(img, frame_num, bboxes, tmp_dir):
    for person_id, bbox in bboxes.iteritems():
        bbox = bbox.resize(scale=1.3)
        crop_file_path = os.path.join(tmp_dir, 'frame%dperson%d.jpg' % (frame_num, person_id))
        crop_img = img[bbox.top:bbox.bottom, bbox.left:bbox.right]
        cv2.imwrite(crop_file_path, crop_img)


def extract_faces(video_file_path, frames_limit, tmp_dir, detection_step):
    faces_df = pd.DataFrame(columns=['frame', 'person_id', 'x', 'y', 'w', 'h'])

    input_video = cv2.VideoCapture(video_file_path)
    has_more_frames, frame = input_video.read()

    tracker = Tracker()
    for _ in tqdm(range(frames_limit)):
        if not has_more_frames:
            break
        has_more_frames, cur_frame = input_video.read()

        tracker.read_frame(cur_frame)
        tracker.track_faces()
        if tracker.cur_frame_num % detection_step == 0:
            tracker.match_faces()

        new_bboxes_by_id = tracker.bboxes_by_id
        crop_faces(frame, tracker.cur_frame_num, new_bboxes_by_id, tmp_dir)

        for person_id in new_bboxes_by_id:
            face = new_bboxes_by_id[person_id]
            faces_df = faces_df.append({'frame': tracker.cur_frame_num,
                                        'person_id': person_id,
                                        'x': face.x,
                                        'y': face.y,
                                        'w': face.w,
                                        'h': face.h},
                                       ignore_index=True,)
    return faces_df
