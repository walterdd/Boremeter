import os
import csv

import pandas as pd
import cv2
from tqdm import tqdm

from .detector import detect_faces
from .bounding_boxes import have_intersection, bboxes_are_close


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

    initial_timeout = 50    # number of frames to keep the bbox
    cur_frame_num = 0
    old_bboxes_by_id = {}         # bboxes from the previous frame
    max_id = 0              # last given id
    timeouts = {}           # current timeouts for bboxes

    for cur_frame_num in tqdm(range(frames_limit)):
	if not  has_more_frames:
		break

        while cur_frame_num % detection_step != 0 and has_more_frames:
            has_more_frames, frame = input_video.read()
            cur_frame_num += 1

        if not has_more_frames or cur_frame_num >= frames_limit:
            break

        new_bboxes_by_id = {}
        found_bboxes = detect_faces(frame)  # array of bboxes
        tracked_faces = {}
        face_ids_to_delete = []  # array of ids

        for bbox in found_bboxes:
            new_bboxes_by_id[max_id] = bbox
            timeouts[max_id] = initial_timeout
            max_id += 1

        for new_id, new_bbox in new_bboxes_by_id.iteritems():
            face_found = False
            for old_id, old_bbox in old_bboxes_by_id.iteritems():
                if timeouts[old_id] > 0:
                    if not face_found and bboxes_are_close(old_bbox, new_bbox):
                        tracked_faces[old_id] = new_bbox
                        timeouts[old_id] = initial_timeout
                        face_found = True
                        face_ids_to_delete.append(new_id)
                    elif not face_found and have_intersection(old_bbox, new_bbox):
                        face_ids_to_delete.append(new_id)
                        face_found = True

        for face_id in face_ids_to_delete:
            del new_bboxes_by_id[face_id]

        for tracked_face in tracked_faces:
            new_bboxes_by_id[tracked_face] = tracked_faces[tracked_face]

        for old_id in old_bboxes_by_id:
            if timeouts[old_id] > 0 and old_id not in tracked_faces:
                timeouts[old_id] -= 1
                new_bboxes_by_id[old_id] = old_bboxes_by_id[old_id]

        crop_faces(frame, cur_frame_num, new_bboxes_by_id, tmp_dir)

        for person_id in new_bboxes_by_id:
            face = new_bboxes_by_id[person_id]
            faces_df = faces_df.append({'frame': cur_frame_num,
                                        'person_id': person_id,
                                        'x': face.x,
                                        'y': face.y,
                                        'w': face.w,
                                        'h': face.h},
                                       ignore_index=True,)

        old_bboxes_by_id = new_bboxes_by_id
        cur_frame_num += 1
        has_more_frames, frame = input_video.read()

    return faces_df
