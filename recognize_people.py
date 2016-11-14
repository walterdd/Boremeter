import pandas as pd
import numpy as np
import sys

caffe_root = 'caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe


def load_nets():
    age_net_pretrained='./caffe_models/dex_imdb_wiki.caffemodel'
    age_net_model_file='./caffe_models/age.prototxt'
    age_net = caffe.Classifier(age_net_model_file, age_net_pretrained,
                               channel_swap=(2,1,0),
                               raw_scale=255,
                               image_dims=(256, 256))

    gender_net_pretrained='./caffe_models/gender.caffemodel'
    gender_net_model_file='./caffe_models/gender.prototxt'
    gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                                  channel_swap=(2,1,0),
                                  raw_scale=255,
                                  image_dims=(256, 256))
    return age_net, gender_net


def make_im_name(frame, person_id):
    return "frame%dperson%d.jpg" % (frame, person_id)


def read_data(detected_file='faces.csv'):

    # read table of detected people

    detected_faces = pd.read_csv(detected_file)
    detected_faces['age'] = np.zeros(detected_faces.shape[0])
    detected_faces['gender'] = np.zeros(detected_faces.shape[0])
    return detected_faces


def recognize_people():
    gender_list = ['Female', 'Male']
    age_net, gender_net = load_nets()

    detected_faces = read_data('faces.csv')

    for i in detected_faces.index:
        input_image = caffe.io.load_image('./faces/' +
                                          make_im_name(detected_faces['frame'][i], detected_faces['person_id'][i]))
        detected_faces['age'][i] = age_net.predict([input_image])[0].argmax()
        detected_faces['gender'][i] = gender_list[gender_net.predict([input_image])[0].argmax()]

    detected_faces.to_csv('recognized.csv')
    return



