import os
import csv

import cv2

from detector import detect_faces
from bounding_boxes import have_intersection, are_close


def crop_faces(img, frame_num, bboxes, tmp_dir):
    for person_id in bboxes:
        bbox = bboxes[person_id]
        bbox = bbox.copy()  # copying to avoid changes in the original bbox
        bbox.resize(scale=1.3)

        cv2.imwrite(os.path.join(tmp_dir, "frame%dperson%d.jpg" % (frame_num, person_id)),
                    img[bbox.top: bbox.bottom, bbox.left: bbox.right])


def extract_faces(video_file_path, frames_limit, tmp_dir, detection_step):

    csvfile = open(os.path.join(tmp_dir, 'faces.csv'), 'wb')
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['frame', 'person_id', 'x', 'y', 'w', 'h'])

    input_video = cv2.VideoCapture(video_file_path)
    ret, frame = input_video.read()

    initial_timeout = 50    # number of frames to keep the bbox
    cur_frame_num = 0
    prev_faces = {}         # bboxes from the previous frame
    max_id = 0              # last given id
    timeouts = {}           # current timeouts for bboxes

    while cur_frame_num < frames_limit and ret:

        while cur_frame_num % detection_step != 0 and ret:
            ret, frame = input_video.read()
            cur_frame_num += 1

        if not ret:
            break

        cur_faces = {}
        found_bboxes = detect_faces(frame)
        tmp_faces = {}
        faces_to_delete = []

        for bbox in found_bboxes:
            cur_faces[max_id] = bbox
            timeouts[max_id] = initial_timeout
            max_id += 1

        for cur_id in cur_faces:
            found = 0
            for prev_id in prev_faces:
                if timeouts[prev_id] > 0:
                    if not found and are_close(prev_faces[prev_id], cur_faces[cur_id]):
                        tmp_faces[prev_id] = cur_faces[cur_id]
                        timeouts[prev_id] = initial_timeout
                        found = 1
                        faces_to_delete.append(cur_id)
                    elif have_intersection(prev_faces[prev_id], cur_faces[cur_id]):
                        faces_to_delete.append(cur_id)

        for d in faces_to_delete:
            cur_faces.pop(d, None)

        for tmp in tmp_faces:
            cur_faces[tmp] = tmp_faces[tmp]

        for prev_id in prev_faces:
            if timeouts[prev_id] > 0 and prev_id not in tmp_faces:
                timeouts[prev_id] -= 1
                cur_faces[prev_id] = prev_faces[prev_id].copy()

        crop_faces(frame, cur_frame_num, cur_faces, tmp_dir)
        for person_id in cur_faces:
            x, y, w, h = cur_faces[person_id].get()
            writer.writerow([cur_frame_num, person_id, x, y, w, h])

        prev_faces = cur_faces.copy()
        cur_frame_num += 1
        ret, frame = input_video.read()

    csvfile.close()
