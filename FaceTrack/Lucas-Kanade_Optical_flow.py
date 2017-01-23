from matplotlib import pyplot as plt
import numpy as np

import cv2

rectangle_x = 0


face_classifier = cv2.CascadeClassifier('/home/artem/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')


#cap = cv2.VideoCapture('video/sample.mov')
cap = cv2.VideoCapture(0)


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.01,
                       minDistance = 10,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
restart = True
while restart == True:
    ret, old_frame = cap.read()

    # old_frame = cv2.imread('images/webcam-first-frame-two.png')

    ######Adding my code###
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(old_gray, 1.1, 5)
    #cv2.imshow('Old_Frame', old_frame)
    if len(face) != 0:
        restart = False# print "This is empty"

for (x,y,w,h) in face:
    focused_face = old_frame[y: y+h, x: x+w]
    cv2.rectangle(old_frame, (x,y), (x+w, y+h), (0,255,0),2)

cv2.imshow('Old_Frame', old_frame)
cv2.waitKey(0)
#initalize all pixels to zero (picture completely black)
#print(old_frame.shape)
#print(focused_face.shape)
#mask_use = np.zeros(old_frame.shape,np.uint8)
#print(mask_use.shape)
#Crop old_frame coordinates and paste it on the black mask)
#mask_use[y:y+h,x:x+w] = old_frame[y:y+h,x:x+w]

#height, width, depth = mask_use.shape
#print "Height: ", height
#print "Width: ", width
#print "Depth: ", depth


#height, width, depth = old_frame.shape
#print "Height: ", height
#print "Width: ", width
#print "Depth: ", depth

#cv2.imshow('Stuff', mask_use)

#cv2.imshow('Old_Frame', old_frame)
#cv2.imshow('Zoom in', focused_face)

face_gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)

gray = cv2.cvtColor(focused_face,cv2.COLOR_BGR2GRAY)


#print(mask_use.shape)
#print(old_frame.shape)
mask_use = np.zeros_like(face_gray)
mask_use[y:y+h, x:x+w] = 255
#corners_t = cv2.goodFeaturesToTrack(face_gray, mask = mask_use, **feature_params)
#corners = np.int0(corners_t)

#print corners



#for i in corners:
#    ix,iy = i.ravel()
#    cv2.circle(focused_face,(ix,iy),3,255,-1)
#    cv2.circle(old_frame,(x+ix,y+iy),3,255,-1)

    #print ix, " ", iy

#plt.imshow(old_frame),plt.show()
"""
print "X: ", x
print "Y: ", y
print "W: ", w
print "H: ", h
#face_array = [x,y,w,h]
"""

#############################
p0 = cv2.goodFeaturesToTrack(face_gray, mask = mask_use, **feature_params)
#############################
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
mask_use[y:y+h, x:x+w] = 255
#p0 = (600, 400)
#p0 = np.array([[[255, 428]]])
#p0 = np.array([[[350, 350]]])
# p0[0]
#p0 = np.array([[[385 ,270]]])
#print p0
#print type(p0)
while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    cd = cv2.minAreaRect(p1)
    cd = cv2.boxPoints(cd)
    #print cd
    cd = np.int0(cd)
    #print c
    #print p1 , st, err
    # Select good points
    good_new = p1
    ###print "Good_New"
    ###print good_new
    good_old = p0

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        #print i
        #print color[i]
        a,b = new.ravel()
        c,d = old.ravel()
        #cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        cv2.circle(frame,(a, b),5,color[i].tolist(),-1)
        #cd = cv2.boundingRect(cd)
        #print cd
        if i == 99:
            break
        #For circle, maybe replace (a,b) with (c,d)?
    #img = cv2.add(frame,mask)
    cv2.polylines(frame, [cd], True, 255, 2)
    cd = cv2.boundingRect(cd)
    (x, y, w, h) = cd
    cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
    print cd
    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)





cv2.destroyAllWindows()
cap.release()