import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('/home/artem/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
#eye_cascade = cv2.CascadeClassifier('/home/artem/anaconda3/envs/YDF/share/OpenCV/haarcascades/haarcascade_eye.xml')

cap = cv2.VideoCapture('/home/artem/grive/HSE/3course/YDF_proj/vid/Vid/dow480.mp4')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        #for (ex, ey, ew, eh) in eyes:
        #    cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (0,255,0), 2)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()