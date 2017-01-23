import numpy as np
import cv2
import csv
import math
from matplotlib import pyplot as plt

# bb = bounding box
color = np.random.randint(0,255,(100,3))

"""
Section of bounding boxes operations

"""

def scale_bb(x, y, w, h, max_x, max_y, scale):

    # returns scaled parameters of the bounding box

    return [
            int(x - w * (scale - 1)/2),
            int(y - h * (scale - 1)/2),
            int(w * (scale)),
            int(h * (scale))
            ]

def pts_to_bb(pts):
    x = min(pts[0][0], pts[1][0], pts[2][0], pts[3][0])
    y = min(pts[0][1], pts[1][1], pts[2][1], pts[3][1])
    w = max(pts[0][0], pts[1][0], pts[2][0], pts[3][0]) - x
    h = max(pts[0][1], pts[1][1], pts[2][1], pts[3][1]) - y
    return [x, y, w, h]

def get_bbs(img, detector):

    # returns all found faces coordinates on the image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return detector.detectMultiScale(gray, 1.1, 5)


def have_intersection(bb1, bb2):

    # check if two bounding boxes have an intersection

    return not (bb1[0] + bb1[2] < bb2[0]
                or bb2[0] + bb2[2] < bb1[0]
                or bb1[1] + bb1[3] < bb2[1]
                or bb2[1] + bb2[3] < bb1[1])


def are_close(bb1, bb2):

    # check if two bounding boxes are close and have similar shapes

    if abs(bb1[2]*bb1[3] - bb2[2]*bb2[3]) > max(bb1[2]*bb1[3], bb2[2]*bb2[3]) / 4:
        return False

    return (abs(bb1[0] + bb1[2] / 2 - bb2[0] - bb2[2] / 2) < bb1[2] / 2) and \
            (abs(bb1[1] + bb1[3] / 2 - bb2[1] - bb2[3] / 2) <  bb1[3] / 2)


"""
Editing found faces

"""
def track_obj(face, frame):
    track_window = face
    (x, y, w, h) = face
    (x, y, w, h) = (int(max(0, x)), int(max(0, y)), int(max(0, w)), int(max(0, h)))
    roi = frame[y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 51., 89.)), np.array((17., 140., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    return track_window, roi, hsv_roi, mask, roi_hist, term_crit



def preprocess_bbs(bbs, frames_arr, test_fl, timeout=200, im_width=800, im_height=400):

    # returns improved bounding boxes with person_id


    faces = {} # faces dict before proccessing
    max_id = 1

    for fn in bbs:
        faces[fn] = {}
        for bb in bbs[fn]:
            faces[fn][max_id] = {}
            faces[fn][max_id]['timeout'] = timeout
            faces[fn][max_id]['coords'] = scale_bb(bb[0], bb[1], bb[2], bb[3], im_width, im_height, 1.3)
            max_id += 1


    new_faces = {} # faces dict after proccessing

    new_faces[0] = faces[0]

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    opt_flow_dict = {}
    mean_shift_dict = {}

    detection_cell = 10

    for frame in faces:
        if frame != 0:
            new_faces[frame] = {}

            # update detected faces

            if frame % detection_cell == 0: # detect faces each detection_cell frame
                for cur_id in faces[frame]:
                    found = 0
                    intersect = 0
                    for prev_id in new_faces[frame - 1]:
                        if (have_intersection(new_faces[frame - 1][prev_id]['coords'],
                                              faces[frame][cur_id]['coords'])):
                            intersect = 1

                        if new_faces[frame - 1][prev_id]['timeout'] > 0:
                            if not found and are_close(new_faces[frame - 1][prev_id]['coords'],
                                                       faces[frame][cur_id]['coords']):

                                new_faces[frame][prev_id] = faces[frame][cur_id].copy()
                                new_faces[frame - 1][prev_id]['timeout'] = -1
                                new_faces[frame][prev_id]['timeout'] = timeout
                                found = 1

                    if not found and not intersect:
                        # insert new face which was not detected before

                        new_faces[frame][cur_id] = faces[frame][cur_id]

                        # track this face for meanshift

                        track_windows, rois, hsv_rois, masks, roi_hists, term_crit = \
                                track_obj(new_faces[frame][cur_id]['coords'], frames_arr[frame])
                        mean_shift_dict[cur_id] = [track_windows, roi_hists]

                        # track this face for optical flow

                        (x, y, h, w) = new_faces[frame][cur_id]['coords']

                        old_gray = cv2.cvtColor(frames_arr[frame], cv2.COLOR_BGR2GRAY)
                        mask = np.zeros_like(old_gray)
                        mask[(y+h/4):(y+3*h/4), (x+w/4):(x+3*w/4)] = 1
                        p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask, **feature_params)
                        opt_flow_dict[cur_id] = [old_gray, p0]
                        #for i in p0:
                        #    ix, iy = i.ravel()
                        #    cv2.circle(focused_face, (ix, iy), 3, 255, -1)
                        #    cv2.circle(old_frame, (x + ix, y + iy), 3, 255, -1)

            # update lost faces from previous frame

            for prev_id in new_faces[frame - 1]:
                if new_faces[frame - 1][prev_id]['timeout'] > 0:
                    new_faces[frame - 1][prev_id]['timeout'] -= 1

                    if not prev_id in opt_flow_dict: # start tracking

                        # MEANSHIFT
                        if test_fl == 1 or test_fl == 3:
                            track_windows, rois, hsv_rois, masks, roi_hists, term_crit = \
                                track_obj(new_faces[frame - 1][prev_id]['coords'], frames_arr[frame-1])

                            mean_shift_dict[prev_id] = [track_windows, roi_hists]

                        # OPTICAL FLOW
                        if test_fl == 2 or test_fl == 3:
                            (x, y, h, w) = new_faces[frame - 1][prev_id]['coords']

                            old_gray = cv2.cvtColor(frames_arr[frame - 1], cv2.COLOR_BGR2GRAY)

                            mask = np.zeros_like(old_gray)
                            mask[(y+h/4):(y+3*h/4), (x+w/4):(x+3*w/4)] = 1

                            p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask, **feature_params)

                            opt_flow_dict[prev_id] = [old_gray, p0]

                        # update bb
                        new_faces[frame][prev_id] = new_faces[frame - 1][prev_id].copy()


                    else: # continue tracking
                        (x, y, w, h) = (0, 0, 0, 0)
                        (x2, y2, w2, h2) = (0, 0, 0, 0)
                        # OPTICAL FLOW
                        if test_fl == 2 or test_fl == 3:
                            p0 = opt_flow_dict[prev_id][1]
                            old_gray = opt_flow_dict[prev_id][0]

                            frame_gray = cv2.cvtColor(frames_arr[frame], cv2.COLOR_BGR2GRAY)

                            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                            good_new = p1[st==1]
                            good_old = p0[st==1]

                        # create new bb coordinates for optical flow from good_new


                        # i do very stupid , take the crainii points and build bb on them

                            x = 2000  # big numbers
                            y = 2000

                            for p in good_new:
                                x = min(x, p[0])
                                y = min(y, p[1])

                            w = 0
                            h = 0

                            for p in good_new:
                                w = max(w, p[0] - x)
                                h = max(h, p[1] - y)

                            # Draw points
                            for i, (new, old) in enumerate(zip(good_new, good_old)):
                                # print i
                                # print color[i]
                                a, b = new.ravel()
                                c, d = old.ravel()
                                cv2.circle(frames_arr[frame], (a, b), 2, color[i].tolist(), -1)
                                if i == 99:
                                    break

                            cv2.imshow('frame', frames_arr[frame])
                            k = cv2.waitKey(30) & 0xff
                            if k == 27:
                                break
                        # update tracking dict
                            old_gray = frame_gray.copy()
                            p0 = good_new.reshape(-1,1,2)
                            opt_flow_dict[prev_id] = [old_gray, p0]

                        # MEANSHIFT
                        if test_fl == 1 or test_fl == 3:
                            hsv = cv2.cvtColor(frames_arr[frame], cv2.COLOR_BGR2HSV)
                            dst = cv2.calcBackProject([hsv], [0], mean_shift_dict[prev_id][1], [0, 180], 1)
                            ret, mean_shift_dict[prev_id][0] = cv2.meanShift(dst, tuple(mean_shift_dict[prev_id][0]), term_crit)
                            (x2, y2, w2, h2) = mean_shift_dict[prev_id][0]

                        # update tracking dict
                        new_faces[frame][prev_id] = new_faces[frame - 1][prev_id].copy()

                        # (x, y, w, h) - results optflow
                        # (x2, y2, w2, h2) - results meanshift
                        # i take mean from them and save like a new bb
                        if test_fl == 2:
                            new_faces[frame][prev_id]['coords'] = (int(x), \
                                                                   int(y), \
                                                                   int(w), \
                                                                   int(h))
                        else:
                            new_faces[frame][prev_id]['coords'] = (int((x + x2) / 2), \
                                                                   int((y + y2) / 2), \
                                                                   int((w + w2) / 2), \
                                                                   int((h + h2) / 2))




    return new_faces


"""
Section of reading, drawing and writing

"""

def write_cropped_image_by_bb(folder_path, frame_num, person_id, img, bb):
    cv2.imwrite(folder_path +  "/frame%dperson%d.jpg" % (frame_num, person_id),
                img[bb[1] : bb[1] + bb[3], bb[0] : bb[0] + bb[2]]);


def write_video(input_file, frames, output_file):

    # visualizes bbs at a new video

    vidFile = cv2.VideoCapture(input_file)
    ret, frame = vidFile.read()

    height, width, layers =  frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    output_video = cv2.VideoWriter(output_file, fourcc, 10.0, (width, height))

    raw_faces, frames = frames

    frames = preprocess_bbs(raw_faces, frames, test_fl=3)

    for frame_num in frames:
        ret, frame = vidFile.read()
        output_video.write(draw_faces_bbs(frame, frames[frame_num]))

    output_video.release()



def draw_faces_bbs(img, faces_bbs):

    # draw rectangles with labels on img

    for face_id in faces_bbs:
        (x,y,w,h) = faces_bbs[face_id]['coords']
        cv2.putText(img, str(face_id), (x, y), 1, 1, (0,0,255), 2, cv2.LINE_AA)
        img = draw_rect(img, scale_bb(x, y, w, h, img.shape[0], img.shape[1], 1))

    return img


def draw_rect(img, bb):

    # just draw a rectangle

    x, y, w, h = bb
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255 ,0), 1)
    return img


def video_to_frames_dict(input_file, frames_num, detector):

    # convert video file to dictionary of frames, ids and bbs

    vidFile = cv2.VideoCapture(input_file)
    cur_frame = 0
    frames = {}

    ret = True

    all_frames = [None] * frames_num
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    while cur_frame < frames_num and ret:
        ret, frame = vidFile.read()
        all_frames[cur_frame] = frame
        frames[cur_frame] = get_bbs(frame, detector)
        cur_frame += 1

    return frames, all_frames


def save_dict_as_csv(faces_dictionary):
    with open('faces.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['frame', 'person_id', 'x', 'y', 'w', 'h'])
        for frame_num in faces_dictionary:
            for person_id in faces_dictionary[frame_num]:
                x, y, w, h = faces_dictionary[frame_num][person_id]['coords']
                writer.writerow([frame_num, person_id, x, y, w, h])


def video_to_faces(folder_path, input_video, frames_num, detector):

    raw_faces, frames = video_to_frames_dict(input_video, frames_num, detector)

    faces = preprocess_bbs(raw_faces, frames, test_fl=2)

    save_dict_as_csv(faces)

    vidFile = cv2.VideoCapture(input_video)
    cur_frame = 0
    ret = 1

    while cur_frame < frames_num and ret:
        ret, frame = vidFile.read()
        for person_id in faces[cur_frame]:
            write_cropped_image_by_bb(folder_path, cur_frame, person_id, frame, faces[cur_frame][person_id]['coords'])
        cur_frame += 1

    return

def extract_people(video_file, visualize=False, frames_limit=100):
    face_cascade = cv2.CascadeClassifier('/home/artem/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')

    if visualize:
        write_video(video_file, video_to_frames_dict(video_file, frames_limit, detector=face_cascade), '/home/artem/grive/HSE/3course/YDF_proj/vid/Vid/video_aud_cam.avi')

#     video_to_faces('./faces', video_file, frames_limit, face_cascade)

extract_people('/home/artem/grive/HSE/3course/YDF_proj/vid/Vid/bad_variant.mov', True, 50)