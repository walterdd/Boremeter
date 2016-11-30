import csv
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
    return


def draw_rect(img, bb):
 
    x, y, w, h = bb
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255 ,0), 1)
    return img


def draw_faces_bbs(img, faces_bbs):
    
    for face_id in faces_bbs:
        (x,y,w,h) = faces_bbs[face_id]['coords']
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

    return 

def frame_to_faces(frame, frame_num, folder_path='./cropped'):
    

def save_cropped_faces(frames, folder_path='./cropped'):
    for frame_num in frames:
        for person_id in frames[frame_num][1]:
            bb = frames[frame_num][1][person_id]['coords']
            img = frames[frame_num][0]
            cv2.imwrite(folder_path +  "/frame%dperson%d.jpg" % (frame_num, person_id), 
                        img[bb[1] : bb[1] + bb[3], bb[0] : bb[0] + bb[2]])
    return 


def extract_whole_data(video_file_path, visualize=False, faces_folder='cropped'):

    faces = detect_faces_on_video(video_file_path)
    faces = track_faces(faces)

    save_cropped_faces(faces, faces_folder)
    if visualize:
        visualize_bbs(faces)
    write_faces_to_csv(faces)

    return

def extract_people_from_video(video_file_path, visualize=False, faces_folder='cropped'):
    input_video = cv2.VideoCapture(video_file_path)

    ret, cur_frame_img = input_video.read() 
    cur_frame_num = 1    

    while cur_frame_num < frames_limit and ret:
        ret, frame = input_video.read() 

        if cur_frame % detection_step == 0:
            frames[cur_frame][1] = detect_faces(frame)

        else:
            frames[cur_frame][1] = np.array([])

        cur_frame += 1
    
    return frames