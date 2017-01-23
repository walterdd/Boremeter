#coding=utf-8
import numpy as np
import cv2
import video
from common import anorm2, draw_str
#from time import clock
face_classifier = cv2.CascadeClassifier('/home/artem/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')




# Настраиваем пар-ы для алгоритам Lucas-kanade (оптический поток)
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Настраиваем пар-ы для алгоритма Shi-Thomasi посика хороших точек
feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
#idd = 0

def transform_to_p(a):
    (x, y, w, h) = a
    return np.array([[x, y], [x+w, y], [x+w, y+w], [x, y+w]])


def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return False # or (0,0,0,0) ?
  return (x, y, w, h)


def run(video_src1):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output15.avi', fourcc, 20.0, (1280, 720))
    track_len = 10
    detect_interval = 5
    tracks = [] # текущее трекающие точки
    tracks_bb = []
    #self.cam = video.create_capture(video_src)
    cap = cv2.VideoCapture(video_src1)
    cur_diff = []
    frame_idx = 0 # Индекс тек фрейма
    idd = 0
    while True:
        #ret, frame = self.cam.read()
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()


        if frame_idx % detect_interval == 0:
            faces = face_classifier.detectMultiScale(frame_gray, 1.1, 5)
            idd = 0


            for (x, y, w, h) in faces:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        out.write(vis)
        frame_idx += 1
        prev_gray = frame_gray
        #out.write(vis)
        #cv2.imshow('try', vis)

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    out.release()

def main():
    video_src = '/home/artem/grive/HSE/3course/YDF_proj/vid/Vid/badboys.mov'

    print __doc__
    run(video_src)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()