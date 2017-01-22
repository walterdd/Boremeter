import pandas as pd
import numpy as np
import sys
from detector import *


caffe_root = '../caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe


def load_nets(net_type):
    new_net = ""
    if net_type == 'age':
        age_net_pretrained='../caffe_models/dex_imdb_wiki.caffemodel'
        age_net_model_file='../caffe_models/age.prototxt'
        new_net = caffe.Classifier(age_net_model_file, age_net_pretrained,
                                   channel_swap=(2,1,0),
                                   raw_scale=255,
                                   image_dims=(256, 256))

    if net_type == 'gender':
        gender_net_pretrained='../caffe_models/gender.caffemodel'
        gender_net_model_file='../caffe_models/gender.prototxt'
        new_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                                      channel_swap=(2,1,0),
                                      raw_scale=255,
                                      image_dims=(256, 256))
    return new_net


def make_im_name(frame, person_id):
    return "frame%dperson%d.jpg" % (frame, person_id)


def read_data(detected_file='faces.csv'):

    # read table of detected people

    detected_faces = pd.read_csv(detected_file)
    detected_faces['age'] = np.zeros(detected_faces.shape[0])
    detected_faces['gender'] = np.zeros(detected_faces.shape[0])
    detected_faces['interest'] = np.zeros(detected_faces.shape[0])
    return detected_faces

def recognize_people(frames_limit=10000000, step=30):
    gender_list = ['Female', 'Male']
    age_net = load_nets('age')

    detected_faces = read_data('faces.csv')

    capture_step = np.unique(detected_faces['frame'])[1] - np.unique(detected_faces['frame'])[0]

    for i in detected_faces.index:
        print i, ' of ', detected_faces.shape[0]
        if detected_faces['frame'][i] > frames_limit:
            break
        if (detected_faces['frame'][i] / capture_step) % step == 0:
            input_image = caffe.io.load_image('./cropped/' + \
                                          make_im_name(detected_faces['frame'][i], detected_faces['person_id'][i]))
            detected_faces['age'][i] = age_net.predict([input_image], oversample=False)[0].argmax()
        else:
            detected_faces['age'][i] = 'nan'

    del age_net

    gender_net = load_nets('gender')

    for i in detected_faces.index:
        print i, ' of ', detected_faces.shape[0]
        if detected_faces['frame'][i] > frames_limit:
            break
        if (detected_faces['frame'][i] / capture_step) % step == 0:
            input_image = caffe.io.load_image('./cropped/' + \
                                              make_im_name(detected_faces['frame'][i], detected_faces['person_id'][i]))
            detected_faces['gender'][i] = gender_list[gender_net.predict([input_image], oversample=False)[0].argmax()]
        else:
            detected_faces['gender'][i] = 'nan'


    del gender_net

    cc = cv2.CascadeClassifier(global_config['VJ_cascade_path'])

    for i in detected_faces.index:
        print i, ' of ', detected_faces.shape[0]
        if detected_faces['frame'][i] > frames_limit:
            break
        if detected_faces['frame'][i] % 2 == 0:
            # print 'proc'
            im = cv2.imread('./cropped/' +
                                              make_im_name(detected_faces['frame'][i], detected_faces['person_id'][i]))
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # print cc.detectMultiScale(gray, 1.1, 1)
            detected_faces['interest'][i] = len(cc.detectMultiScale(gray, 1.1, 1)) > 0
        else:
            detected_faces['interest'][i] = -1

    detected_faces.to_csv('recognized.csv')
    return



def get_stats(table):
    import matplotlib.pyplot as plt
    data = pd.read_csv(table)
    rec_data = data[(data['gender'] == 'Male') | (data['gender'] == 'Female')]

    mpc =  float((rec_data['gender'] == 'Male').sum()) / rec_data.shape[0] * 100
    mean_age  rec_data['age'].mean()

    x = data[['frame', 'interest']].groupby('frame').sum() / data[['frame', 'interest']].groupby('frame').size()[0]

    y = x['interest']
    x = x.index / 25 
    return mpc, mean_age, x, y
