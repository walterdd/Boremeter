import numpy as np
import cv2
#from matplotlib import pyplot as plt

face_classifier = cv2.CascadeClassifier('/home/artem/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv2.VideoCapture(0)
# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
Restart = True
while Restart:
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(old_gray, 1.1, 5)
    if len(face) != 0:
        #cv2.imshow('lol', old_frame)
        #cv2.waitKey(0)
        Restart = False

for (x, y, w, h) in face:
    focused_face = old_frame[y: y + h, x: x + w]
    #cv2.rectangle(old_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#Initialize mask
mask = np.zeros(old_frame.shape[:-1], np.uint8)
print(mask.shape)
mask[y:y+h, x:x+w] = old_gray[y:y+h, x:x+w]
p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)
cv2.rectangle(old_frame, (x,y), (x+w, y+h), (0,255,0),2)
#cv2.imshow('of', old_frame)
#cv2.waitKey(0)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    it = 0
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        #print(a,b,c,d)
        #mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        it+=1
    img = cv2.add(frame,mask)
    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()