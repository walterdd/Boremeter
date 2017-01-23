import sys
import cv2
import subprocess
import numpy as np


PATH_VID = '/home/artem/grive/HSE/3course/YDF_proj/vid/Vid/'
PATH_FRAMES = '/home/artem/grive/HSE/3course/YDF_proj/vid/Frames/'
PATH_FFMPEG = '/home/artem/anaconda3/bin/ffmpeg'

def face_detect(path, Cascade):
    return 0


def cut_frames(vid_names, time_s):
    sep_time =  "select='not(mod(t\,%f))'" % time_s
    for name in vid_names:
        img_name = name.split('.')[0]
        frame_pth = PATH_FRAMES + img_name +'/' + img_name + '_%d.png'
        subprocess.call(['mkdir', PATH_FRAMES + img_name])
        subprocess.call([PATH_FFMPEG, '-i', PATH_VID + name, '-vf' , sep_time, '-vsync', 'vfr', frame_pth])
        #print('-i '+ PATH_VID + name+' -vf '+ sep_time+ ' -vsync '+ ' vfr '+ PATH_FRAMES + name + '/' +'output_%d.png')
        #subprocess.call(['/home/artem/anaconda3/bin/ffmpeg', '-i',  '/home/artem/grive/HSE/3course/YDF_proj/vid/Vid/Putin.mp4'])
        #print(sep_time)
    return 0
#imagePath = sys.argv[1]
#cascPath = sys.argv[2]
#imagePath = '/home/artem/grive/HSE/3course/YDF_proj/vid/Frames/1.png'

#face_cascade = cv2.CascadeClassifier('/home/artem/anaconda3/envs/YDF/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
#image = cv2.imread(imagePath)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#faces = face_cascade.detectMultiScale(
#    gray,
#    scaleFactor=1.1,
#    minNeighbors=5,
 #   minSize=(30, 30),
#    flags = cv2.CASCADE_SCALE_IMAGE
#)

# Draw a rectangle around the faces
#imgs =[]
#for (x, y, w, h) in faces:
#    #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#    newimage = image[y: y + h, x: x + w]
#    imgs.append(newimage)
#cv2.imshow("Face found" ,image)
#for i in range(len(imgs)):
#    cv2.imwrite('Putin_face%d.jpg' % (6+i), imgs[i])
#cv2.waitKey(0)