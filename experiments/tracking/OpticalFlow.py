#coding=utf-8
import numpy as np
import cv2
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
    out = cv2.VideoWriter('output13.avi', fourcc, 20.0, (1280, 720))
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


        FaceFind = True
        #if len(self.tracks) > 0 and FaceFind:
        for i in range(len(tracks)):
            if len(tracks[i]) > 0 and FaceFind:
                img0, img1 = prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in tracks[i]]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

                d = abs(p0-p0r).max(-1)

                #print d

                good = d < 1
                new_tracks = []
                #g_p = []
                for tr, (x, y), good_flag in zip(tracks[i], p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))

                    if len(tr) > track_len:
                        del tr[0]
                    new_tracks.append(tr)
                #    g_p.append([x, y])
                    #cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                #g_p = np.array(g_p)
                #g_p = np.mean(g_p, axis=0)
                #print g_p
                #if type(g_p) is not np.float64:
                #    print g_p
                #    cv2.circle(vis, (g_p[0], g_p[1]), 2, (0, 255, 0), -1)
                #print self.tracks[i]
                #print new_tracks
                tracks[i] = new_tracks
                center_m = []
                for z in tracks[i]:
                    for l in z:
                        center_m.append([l])
                center_m = np.mean(center_m, axis=0)
                #if len(center_m) > 0:
                if type(center_m) is not np.float64:
                    (x1, y1, w1, h1) = tracks_bb[i]
                    tracks_bb[i] = (int(center_m[0][0]+cur_diff[i][0]), int(center_m[0][1]+cur_diff[i][1]), w1, h1)
                    cv2.circle(vis, (center_m[0][0], center_m[0][1]), 2, (255, 0, 0), -1)
                #self.tracks[i] = tr
                cd = cv2.minAreaRect(p1)
                cd = cv2.boxPoints(cd)
                # print cd
                cd = np.int0(cd)
                #if i == 1:
                #cv2.polylines(vis, [cd], True, 255, 2)
                #cv2.polylines(vis, [np.int32(tr) for tr in self.tracks[i]], False, (0, 255, 0))
        if frame_idx % detect_interval == 0:
            faces = face_classifier.detectMultiScale(frame_gray, 1.1, 5)
            idd = 0
            for (x, y, w, h) in faces:
                #print faces[0]
                find_id = -1
                #bb = cv2.boundingRect(np.array([[x, y], [x + w, y + h]]))
                #cv2.Rect
                #print type(cv2.boundingRect(np.array([[x, y], [x + w, y + h]])))
                #print type(bb)
                for j in range(len(tracks_bb)):
                    inter_coo = intersection(tracks_bb[j], (x, y, w, h))
                    if inter_coo:
                        s_int = cv2.contourArea(transform_to_p(inter_coo))
                        s1 = cv2.contourArea(transform_to_p((x, y, w, h)))
                        s2 = cv2.contourArea(transform_to_p(tracks_bb[j]))
                        if ((s1/s_int) < 2) or ((s2/s_int) < 2):
                            tracks_bb[j] = (x, y, w, h)
                            find_id = j
                            break

                if find_id == -1:
                    tracks_bb.append((x, y, w, h))
                    tracks.append([])
                    cur_diff.append([])
                    find_id = len(tracks)-1
                #print(1)
                #cv2.imshow('ol', frame_gray[y:y+h, x:x+w])
                #cv2.waitKey(0)
                mask = np.zeros_like(frame_gray)
                mask[(y+h/4):(y+3*h/4), (x+w/4):(x+3*w/4)] = 255
                (x1, y1, w1, h1) = (x, y, w, h)
                #a = cv2.boundingRect(np.array([[x, y], [x+w, y+h]]))
                #z1, z2, z3, z4 = a
                #cv2.rectangle(vis, (z1, z2), (z1 + z3, z2 + z4), (0, 255, 0), 2)
                #cv2.rectangle(vis, (x+w/4, y+h/4), (x + 3*w/4, y + 3*h/4), 255, 2)
                for (x, y) in [np.int32(tr[-1]) for tr in tracks[find_id]]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        tracks[find_id].append([(x, y)])
                #print np.mean(self.tracks, axis=0)
                #try:
                #    print np.mean(self.tracks[find_id], axis=0)
                #except TypeError:
                #    while(True):
                center_m = []
                for z in tracks[find_id]:
                    for l in z:
                        center_m.append([l])
                center_m = np.mean(center_m, axis=0)
                cv2.circle(vis, (center_m[0][0], center_m[0][1]), 2, (0, 0, 255), -1)
                if type(center_m) is not np.float64:
                    cur_diff[find_id] = [x1-center_m[0][0], y1-center_m[0][1]]
                    #print self.cur_diff[find_id]
                #print center_m[0][0]
                idd += 1

        for (x, y, w, h) in tracks_bb:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        frame_idx += 1
        prev_gray = frame_gray
        out.write(vis)
        #cv2.imshow('try', vis)

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    out.release()

def main():
    video_src = '/home/artem/grive/HSE/3course/YDF_proj/vid/Vid/badboys.mov'
    run(video_src)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()