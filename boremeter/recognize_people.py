import os
import gc

import pandas as pd
import numpy as np
import cv2
import caffe
from tqdm import tqdm

from .detector import DETECTOR_CONFIG


class FaceRecognizer:
    def __init__(self, models_folder, caffe_model, prototype):
        model = os.path.join(models_folder, prototype)
        pretrained_weights = os.path.join(models_folder, caffe_model)

        self.net = caffe.Classifier(
            model,
            pretrained_weights,
            channel_swap=(2, 1, 0),
            raw_scale=255,
            image_dims=(256, 256),
        )

    def __del__(self):
        del self.net

    def predict(self, x):
        return self.net.predict([x], oversample=False)


class AgeRecognizer(FaceRecognizer):
    def predict(self, x):
        return FaceRecognizer.predict(self, x)[0].argmax()


class GenderRecognizer(FaceRecognizer):
    def predict(self, x):
        genders = ['Female', 'Male']
        return genders[FaceRecognizer.predict(self, x)[0].argmax()]


def make_image_name(frame_num, person_id):
    return 'frame%dperson%d.jpg' % (frame_num, person_id)


def recognize_faces(detected_faces, tmp_dir, frames_limit, caffe_models_path, recognition_step):
    # load pre-trained nets

    age_recognizer = AgeRecognizer(caffe_models_path, 'age.caffemodel', 'age.prototxt')

    # read table of detected people
    # populate age, gender and interest with zeros for the moment

    # detected_faces = pd.read_csv(os.path.join(tmp_dir, 'faces.csv'))
    detected_faces.loc[:, 'age'] = np.zeros(detected_faces.shape[0])
    detected_faces.loc[:, 'gender'] = np.zeros(detected_faces.shape[0])
    detected_faces.loc[:, 'interest'] = np.zeros(detected_faces.shape[0])

    ages = {}

    # recognize age if frame_id % recognition_step == 0

    for i, face_row in tqdm(detected_faces.iterrows()):
        if face_row['frame'] > frames_limit:
            break
        if face_row['frame'] % recognition_step == 0:
            im_name = make_image_name(face_row['frame'], face_row['person_id'])
            full_file_name = os.path.join(tmp_dir, im_name)
            input_image = caffe.io.load_image(full_file_name)

            detected_faces.loc[i, 'age'] = age_recognizer.predict(input_image)

            if face_row['person_id'] in ages:
                ages[face_row['person_id']][0] += face_row['age']
                ages[face_row['person_id']][1] += 1
            else:
                ages[face_row['person_id']] = [face_row['age'], 1]

    del age_recognizer  # deleting net to clean the memory
    gc.collect()

    for i, face_row in detected_faces.iterrows():

        try:
            ages_sum, count = ages[face_row['person_id']]
            detected_faces.loc[i, 'age'] = ages_sum / count

        except KeyError:
            detected_faces.loc[i, 'age'] = 25  # mean? median?

    # recognize gender if frame_id % recognition_step == 0

    gender_recognizer = GenderRecognizer(caffe_models_path, 'gender.caffemodel', 'gender.prototxt')
    genders = {}

    for i, face_row in tqdm(detected_faces.iterrows()):
        if face_row['frame'] > frames_limit:
            break
        if face_row['frame'] % recognition_step == 0:
            im_name = make_image_name(face_row['frame'], face_row['person_id'])
            full_file_name = os.path.join(tmp_dir, im_name)
            input_image = caffe.io.load_image(full_file_name)

            face_row['gender'] = gender_recognizer.predict(input_image)
            if face_row['person_id'] in genders:
                genders[face_row['person_id']][0] += int(face_row['gender'] == 'Male')
                genders[face_row['person_id']][1] += 1
            else:
                genders[face_row['person_id']] = [int(face_row['gender'] == 'Male'), 1]

    del gender_recognizer  # deleting net to clean the memory
    gc.collect()

    for i, face_row in detected_faces.iterrows():

        try:
            detected_faces.loc[i, 'gender'] = 'Male' if (float(genders[face_row['person_id']][0]) /
                                                                 genders[face_row['person_id']][1]) > 0.5 else 'Female'
        except KeyError:
            detected_faces.loc[i, 'gender'] = 'Male'

    face_detector = cv2.CascadeClassifier(DETECTOR_CONFIG['VJ_cascade_path'])

    for i, face_row in tqdm(detected_faces.iterrows()):
        if face_row['frame'] >= frames_limit:
            break

        im = cv2.imread(os.path.join(tmp_dir, make_image_name(face_row['frame'], face_row['person_id'])))
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        detected_faces.loc[i, 'interest'] = len(face_detector.detectMultiScale(gray, 1.1, 1)) > 0

    detected_faces = detected_faces.fillna(np.nan)
    return detected_faces


def get_stats(detected_faces):
    recognized_faces = detected_faces[(detected_faces['gender'] == 'Male') | (detected_faces['gender'] == 'Female')]
    men_pc = float((recognized_faces['gender'] == 'Male').sum()) / recognized_faces.shape[0]
    ages = recognized_faces['age']

    # percentage of interested faces in frames
    interested_pc = (np.array(detected_faces[['frame', 'interest']].groupby('frame').sum()['interest'], dtype=float) /
                     np.array(detected_faces[['frame', 'interest']].groupby('frame').size()) * 100)

    frames_id = np.unique(detected_faces['frame'])
    return men_pc, ages, frames_id, interested_pc
