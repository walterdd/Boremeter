import numpy as np
import cv2
#import pandas as pd
import os
import sys


image_paths = ['/home/artem/grive/HSE/3course/YDF_proj/vid/Frames/Putin/Putin_935.png']
face_cascade = cv2.CascadeClassifier('/home/artem/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
img_num = 0
for imagePath in image_paths:
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        #flags = cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        newimage = image[y: y + h, x: x + w]
        #df2.append([img_num, x, y, w, h])
        cv2.imshow("Face found" ,image)
        #cv2.imwrite('/home/artem/grive/HSE/3course/YDF_proj/faces/face_%d.jpg' % (img_num), newimage)
        img_num += 1
#result = pd.concat(df_im, df1)
#df1.to_csv(df)
cv2.waitKey(0)
