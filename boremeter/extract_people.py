import csv
import cv2
import os
from detector import *
from bbs import *


def crop_faces(img, frame_num, bbs, tmp_dir):
    for person_id in bbs:
        bb = bbs[person_id]
        bb = bb.copy()  # copying to avoid changes in the original bb
        bb.resize(scale=1.3)

        cv2.imwrite(os.path.join(tmp_dir, "frame%dperson%d.jpg" % (frame_num, person_id)),
                    img[bb.top: bb.bottom, bb.left: bb.right])


def extract_faces(video_file_path, frames_limit, tmp_dir, detection_step):

    csvfile = open(os.path.join(tmp_dir, 'faces.csv'), 'wb')
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['frame', 'person_id', 'x', 'y', 'w', 'h'])

    input_video = cv2.VideoCapture(video_file_path)
    ret, frame = input_video.read()

    initial_timeout = 50    # number of frames to keep the bb
    cur_frame_num = 0
    prev_faces = {}         # bbs from the previous frame
    max_id = 0              # last given id
    timeouts = {}           # current timeouts for bbs

    while cur_frame_num < frames_limit and ret:

        while cur_frame_num % detection_step != 0 and ret:
            ret, frame = input_video.read()
            cur_frame_num += 1

        if not ret:
            break

        cur_faces = {}
        found_bbs = detect_faces(frame)
        tmp_faces = {}
        faces_to_delete = []

        for bb in found_bbs:
            cur_faces[max_id] = bb
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
