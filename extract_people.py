
# coding: utf-8

# In[1]:

import cv2
import csv


# In[2]:

# bb = bounding box


"""
Section of bounding boxes operations

"""

def scale_bb(x, y, w, h, max_x, max_y, scale):
    
    # returns scaled parameters of the bounding box
    
    return [
            int(x - w * (scale - 1)/2), 
            int(y - h * (scale - 1)/2), 
            int(w * scale),
            int(h * scale)
            ]


def get_bbs(img, detector):
    
    # returns all found faces coordinates on the image
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return detector.detectMultiScale(gray, 1.3, 5)


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
    

def preprocess_bbs(bbs, timeout=200, im_width=800, im_height=400):
    
    # returns improved bounding boxes with person_id
    
    faces = {}
    max_id = 1

    for fn in bbs:
        faces[fn] = {}
        for bb in bbs[fn]:
            faces[fn][max_id] = {}
            faces[fn][max_id]['timeout'] = timeout
            faces[fn][max_id]['coords'] = scale_bb(bb[0], bb[1], bb[2], bb[3], im_width, im_height, 1.3)
            max_id += 1
            
    
    new_faces = {}
    
    new_faces[0] = faces[0]
    
    for frame in faces:
        if frame != 0:
            new_faces[frame] = {}
            
            for bb in faces[frame]:
                found = 0
                intersect = 0
                for prev_id in new_faces[frame - 1]:          
                    if have_intersection(new_faces[frame - 1][prev_id]['coords'],
                                         faces[frame][bb]['coords']):
                        intersect = 1
                        
                    if new_faces[frame - 1][prev_id]['timeout'] > 0:
                        if not found and are_close(new_faces[frame - 1][prev_id]['coords'], 
                                                   faces[frame][bb]['coords']):
                            new_faces[frame][prev_id] = faces[frame][bb]
                            new_faces[frame - 1][prev_id]['timeout'] = -1
                            new_faces[frame][prev_id]['timeout'] = timeout
                            found = 1
                
                if not found and not intersect:
                    new_faces[frame][bb] = faces[frame][bb]
                            
            for prev_id in new_faces[frame - 1]:
                if new_faces[frame - 1][prev_id]['timeout'] > 0:
                    new_faces[frame - 1][prev_id]['timeout'] -= 1
                    new_faces[frame][prev_id] = new_faces[frame - 1][prev_id]
            
    return new_faces


"""
Section of reading, drawing and writing

"""

def write_cropped_image_by_bb(folder_path, frame_num, person_id, img, bb):
    cv2.imwrite(folder_path + "/frame%dperson%d.jpg" % (frame_num, person_id),
                img[bb[1] : bb[1] + bb[3], bb[0] : bb[0] + bb[2]])


def write_video(input_file, frames, output_file):
    
    # visualizes bbs at a new video
    
    vidFile = cv2.VideoCapture(input_file)
    ret, frame = vidFile.read()
    
    height, width, layers =  frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    output_video = cv2.VideoWriter(output_file, fourcc, 25.0, (width, height))
    
    frames = preprocess_bbs(frames)
    
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
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255 ,0), 2)
    return img


def video_to_frames_dict(input_file, frames_num, detector):
    
    # convert video file to dictionary of frames, ids and bbs
    
    vidFile = cv2.VideoCapture(input_file)
    cur_frame = 0    
    frames = {}
    
    while cur_frame < frames_num:
        ret, frame = vidFile.read() 
        frames[cur_frame] = get_bbs(frame, detector)
        cur_frame += 1
    
    return frames


def save_dict_as_csv(faces_dictionary):
    with open('faces.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['frame', 'person_id', 'x', 'y', 'w', 'h'])
        for frame_num in faces_dictionary:
            for person_id in faces_dictionary[frame_num]:
                x, y, w, h = faces_dictionary[frame_num][person_id]['coords']
                writer.writerow([frame_num, person_id, x, y, w, h])
        

def video_to_faces(folder_path, input_video, frames_num, detector):
    
    faces = preprocess_bbs(video_to_frames_dict(input_video, frames_num, detector))
    
    save_dict_as_csv(faces)
    
    vidFile = cv2.VideoCapture(input_video)
    cur_frame = 0       
    ret = 1
    
    while cur_frame < frames_num and ret:
        ret, frame = vidFile.read() 
        for person_id in faces[cur_frame]:
            write_cropped_image_by_bb(folder_path, cur_frame, person_id, frame,
                                      faces[cur_frame][person_id]['coords'])
        cur_frame += 1
    
    return


def extract_people(video_file, visualize=False, frames_limit=100):
    face_cascade = cv2.CascadeClassifier('./cv_haar_cascades/haarcascade_frontalface_default.xml')
    
    if visualize:
        write_video(video_file, video_to_frames_dict(video_file, frames_limit,
                                                     detector=face_cascade), 'video.avi')
    
    video_to_faces('./faces', video_file, frames_limit, face_cascade)

