
import os
import argparse

from jinja2 import Environment, PackageLoader, select_autoescape

import extract_people
import recognize_people
from visualize import visualize
from util import temporary_directory

DETECTION_STEP = 3
RECOGNITION_STEP = DETECTION_STEP * 6


def gen_html(filename, men_pc, ages, time_arr, attention_arr):
    j2_env = Environment(
        loader=PackageLoader('boremeter', 'templates'),
        autoescape=select_autoescape(['html']),
    )

    template = j2_env.get_template('report.html')

    with open(filename, 'wb') as fh:
        html_report = template.render(
            men_pc=men_pc,
            ages=str(ages.tolist()),
            time_arr=str(time_arr.tolist()),
            attention_arr=str(attention_arr.tolist()),
        )
        fh.write(html_report)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--file', type=argparse.FileType('r'), help='input video file', required=True)
    parser.add_argument('--frames_limit', default=200000, type=int, help='limit of frames to process > 2')
    parser.add_argument('--output_html', default='report.html', type=argparse.FileType('w'),
                        help='path to output .html file with report')
    parser.add_argument('--output_video', default=None, type=argparse.FileType('w'),
                        help='path to output .avi file with visualisation of bounding boxes')
    parser.add_argument('--output_csv', default='recognized.csv', type=argparse.FileType('w'),
                        help='path to output table with information about all detected faces')
    parser.add_argument('--caffe_models_path', default='/root/caffe/models', type=str,
                        help='path to directory with pre-trained caffe models')

    args = parser.parse_args()

    if args.frames_limit < 3:
        raise argparse.ArgumentTypeError('minimum frames_limit is 3')

    caffe_models_path = os.environ.get('CAFFE_MODELS_PATH') or args.caffe_models_path

    # create temporary directory in the current directory where cropped faces will be stored
    with temporary_directory() as tmp_dir:

        print ('Extracting people.....')
        extracted_faces = extract_people.extract_faces(args.file.name, frames_limit=args.frames_limit,
                                     tmp_dir=tmp_dir, detection_step=DETECTION_STEP)

        print ('Extracting statistics.....')
        detected_faces_df = recognize_people.recognize_people(detected_faces=extracted_faces,
                                                              tmp_dir=tmp_dir,
                                                              frames_limit=args.frames_limit,
                                                              caffe_models_path=caffe_models_path,
                                                              recognition_step=RECOGNITION_STEP,)

        detected_faces_df.to_csv(args.output_csv.name)

        print ('Generating html.....')
        men_pc, ages, frames_id, attention_pc = recognize_people.get_stats(detected_faces_df)
        gen_html(args.output_html.name, men_pc, ages, frames_id, attention_pc)

        if args.output_video is not None:
            print ('Visualizing.....')
            visualize(detected_faces_df, args.file.name, args.output_video.name, frames_limit=args.frames_limit,
                      detection_step=DETECTION_STEP)


if __name__ == '__main__':
    main()
