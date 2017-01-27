import csv
import os
from detector import *
from tracker import *

def write_faces_to_csv(frames, output_file='faces.csv'):

    with open('faces.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['frame', 'person_id', 'x', 'y', 'w', 'h'])
        for frame_num in frames:

            for person_id in frames[frame_num][1]:
                x, y, w, h = frames[frame_num][1][person_id]['coords']
                writer.writerow([frame_num, person_id, x, y, w, h])

    csvfile.close()


def draw_rect(img, bb):
 
    x, y, w, h = bb
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255 ,0), 1)
    return img


def draw_faces_bbs(img, faces_bbs):
    
    for face_id in faces_bbs:
        (x,y,w,h) = faces_bbs[face_id]
        cv2.putText(img, str(face_id), (x, y), 1, 1, (0,0,255), 2, cv2.LINE_AA)
        img = draw_rect(img, scale_bb(x, y, w, h, 1))
    
    return img


def visualize_bbs(frames, output_file='visual.avi', fps=25.0):
    
    height, width = frames[0][0].shape[0], frames[0][0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    output_video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    for frame_num in frames:   
        output_video.write(draw_faces_bbs(frames[frame_num][0], frames[frame_num][1]))
    
    output_video.release()


def save_cropped_faces(frames, tmp_dir):
    for frame_num in frames:
        for person_id in frames[frame_num][1]:
            bb = frames[frame_num][1][person_id]['coords']
            img = frames[frame_num][0]
            cv2.imwrite(os.path.join(tmp_dir, "/frame%dperson%d.jpg" % (frame_num, person_id), 
                        img[bb[1] : bb[1] + bb[3], bb[0] : bb[0] + bb[2]]))


def save_cropped_faces_one_frame(img, frame_num, frame_bbs, tmp_dir):
    for person_id in frame_bbs:
        bb = frame_bbs[person_id]
        cv2.imwrite(os.path.join(tmp_dir, "/frame%dperson%d.jpg" % (frame_num, person_id), 
                    img[bb[1] : bb[1] + bb[3], bb[0] : bb[0] + bb[2]]))


def extract_whole_data(video_file_path, visualize, tmp_dir):

    faces = detect_faces_on_video(video_file_path)
    faces = track_faces(faces)

    save_cropped_faces(faces, tmp_dir)
    if visualize:
        visualize_bbs(faces)
    write_faces_to_csv(faces)


def fast_extract(video_file_path, visualize, frames_limit, tmp_dir, detection_step, output_videofile=None):
    input_video = cv2.VideoCapture(video_file_path)

    csvfile =  open(os.path.join(tmp_dir, 'faces.csv'), 'wb')
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['frame', 'person_id', 'x', 'y', 'w', 'h'])

    initial_timeout = 400

    ret, frame = input_video.read() 
    cur_frame_num = 0
    prev_faces = {}
    cur_faces = {}

    max_id = 0

    timeouts = {}

    if visualize:
        height, width = frame.shape[0], frame.shape[1]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        output_video = cv2.VideoWriter(output_videofile.name, fourcc, 25, (width, height))

    while cur_frame_num < frames_limit and ret:

        while cur_frame_num % detection_step != 0:
            ret, frame = input_video.read() 
            cur_frame_num += 1

        if not ret:
            break

        cur_faces = {}

        if cur_frame_num % detection_step == 0:
            found_bbs = detect_faces(frame)

            for bb in found_bbs:
                cur_faces[max_id] = bb
                timeouts[max_id] = initial_timeout
                max_id += 1

        tmp_faces = {}
        faces_to_delete = []

        for cur_id in cur_faces:
            found = 0
            intersect = 0
            for prev_id in prev_faces:      
                if have_intersection(prev_faces[prev_id], cur_faces[cur_id]):
                    intersect = 1
                    faces_to_delete.append(cur_id)

                if timeouts[prev_id] > 0:
                    if not found and are_close(prev_faces[prev_id], cur_faces[cur_id]):
                        tmp_faces[prev_id] = prev_faces[prev_id]
                        timeouts[prev_id] = initial_timeout
                        found = 1
                        faces_to_delete.append(cur_id)

        for d in faces_to_delete:
            cur_faces.pop(d, None)

        for tmp in tmp_faces:
            cur_faces[tmp] = tmp_faces[tmp]

        

        for prev_id in prev_faces:
            if timeouts[prev_id] > 0:
                timeouts[prev_id] -= 1
                cur_faces[prev_id] = prev_faces[prev_id].copy()


        save_cropped_faces_one_frame(frame, cur_frame_num, cur_faces, tmp_dir)

        for person_id in cur_faces:
                x, y, w, h = cur_faces[person_id]
                writer.writerow([cur_frame_num, person_id, x, y, w, h])

        if visualize:
            output_video.write(draw_faces_bbs(frame, cur_faces))

        prev_faces = cur_faces.copy()

        cur_frame_num += 1

        ret, frame = input_video.read() 

    
    csvfile.close()
    
    if visualize:
        output_video.release()

def only_read(file, limit):
    input_video = cv2.VideoCapture(file)
    ret, frame = input_video.read() 
    cn = 0
    while cn < limit and ret:
        ret, frame = input_video.read() 
        cn += 1
