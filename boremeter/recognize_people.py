import os
import gc

import pandas as pd
import numpy as np
import cv2
import caffe
from tqdm import tqdm

from detector import DETECTOR_CONFIG


def load_net(caffe_models_path, net_type):
    new_net = ''
    if net_type == 'age':
        age_net_pretrained = os.path.join(caffe_models_path, 'age.caffemodel')
        age_net_model_file = os.path.join(caffe_models_path, 'age.prototxt')
        new_net = caffe.Classifier(age_net_model_file, age_net_pretrained,
                                   channel_swap=(2, 1, 0),
                                   raw_scale=255,
                                   image_dims=(256, 256))

    if net_type == 'gender':
        gender_net_pretrained = os.path.join(caffe_models_path, 'gender.caffemodel')
        gender_net_model_file = os.path.join(caffe_models_path, 'gender.prototxt')
        new_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                                   channel_swap=(2, 1, 0),
                                   raw_scale=255,
                                   image_dims=(256, 256))
    return new_net


def make_image_name(frame_num, person_id):
    return 'frame%dperson%d.jpg' % (frame_num, person_id)


def recognize_people(tmp_dir, frames_limit, caffe_models_path, recognition_step):
    gender_list = ['Female', 'Male']

    # load pre-trained nets

    age_net = load_net(caffe_models_path, 'age')

    # read table of detected people
    # populate age, gender and interest with zeros for the moment

    detected_faces = pd.read_csv(os.path.join(tmp_dir, 'faces.csv'))
    detected_faces.loc[:, 'age'] = np.zeros(detected_faces.shape[0])
    detected_faces.loc[:, 'gender'] = np.zeros(detected_faces.shape[0])
    detected_faces.loc[:, 'interest'] = np.zeros(detected_faces.shape[0])

    ages = {}

    # recognize age if frame_id % recognition_step == 0

    for i in tqdm(detected_faces.index):
        if detected_faces['frame'][i] > frames_limit:
            break
        if detected_faces['frame'][i] % recognition_step == 0:
            input_image = caffe.io.load_image(os.path.join(tmp_dir,
                                                           make_image_name(detected_faces['frame'][i],
                                                                           detected_faces['person_id'][i])))

            detected_faces.loc[i, 'age'] = age_net.predict([input_image], oversample=False)[0].argmax()

            if detected_faces['person_id'][i] in ages:
                ages[detected_faces['person_id'][i]][0] += detected_faces.loc[i, 'age']
                ages[detected_faces['person_id'][i]][1] += 1
            else:
                ages[detected_faces['person_id'][i]] = [detected_faces.loc[i, 'age'], 1]

    del age_net  # deleting net to clean the memory
    gc.collect()

    for i in detected_faces.index:

        try:
            ages_sum, count = ages[detected_faces['person_id'][i]]
            detected_faces.loc[i, 'age'] = ages_sum / count

        except:
            detected_faces.loc[i, 'age'] = None  # mean? median?

    # recognize gender if frame_id % recognition_step == 0

    gender_net = load_net(caffe_models_path, 'gender')
    genders = {}

    for i in tqdm(detected_faces.index):
        if detected_faces['frame'][i] > frames_limit:
            break
        if detected_faces['frame'][i] % recognition_step == 0:
            input_image = caffe.io.load_image(os.path.join(tmp_dir,
                                                           make_image_name(detected_faces['frame'][i],
                                                                           detected_faces['person_id'][i])))

            detected_faces.loc[i, 'gender'] = gender_list[gender_net.predict([input_image],
                                                                             oversample=False)[0].argmax()]
            if detected_faces['person_id'][i] in genders:
                genders[detected_faces['person_id'][i]][0] += int(detected_faces.loc[i, 'gender'] == 'Male')
                genders[detected_faces['person_id'][i]][1] += 1
            else:
                genders[detected_faces['person_id'][i]] = [int(detected_faces.loc[i, 'gender'] == 'Male'), 1]

    del gender_net # deleting net to clean the memory
    gc.collect()

    for i in tqdm(detected_faces.index):

        try:
            detected_faces.loc[i, 'gender'] = 'Male' if float(genders[detected_faces['person_id'][i]][0]) / \
                                              genders[detected_faces['person_id'][i]][1] > 0.5 else 'Female'
        except:
            detected_faces.loc[i, 'gender'] = 'Male'

    cc = cv2.CascadeClassifier(DETECTOR_CONFIG['VJ_cascade_path'])

    for i in tqdm(detected_faces.index):
        if detected_faces['frame'][i] > frames_limit:
            break
        im = cv2.imread(os.path.join(tmp_dir, make_image_name(detected_faces['frame'][i],
                                                              detected_faces['person_id'][i])))
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        detected_faces.loc[i, 'interest'] = len(cc.detectMultiScale(gray, 1.1, 1)) > 0

    detected_faces = detected_faces.fillna(np.nan)
    return detected_faces


def get_stats(detected_faces):
    recognized_faces = detected_faces[(detected_faces['gender'] == 'Male') | (detected_faces['gender'] == 'Female')]
    men_pc = float((recognized_faces['gender'] == 'Male').sum()) / recognized_faces.shape[0]
    ages = recognized_faces['age']

    # percentage of interested faces in frames
    interested_pc = np.array(detected_faces[['frame', 'interest']].groupby('frame').sum()['interest'], dtype=float) / \
        np.array(detected_faces[['frame', 'interest']].groupby('frame').size()) * 100

    frames_id = np.unique(detected_faces['frame'])
    return men_pc, ages, frames_id, interested_pc
