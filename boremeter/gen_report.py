
import os
import argparse

from jinja2 import Environment, PackageLoader, select_autoescape
import caffe

from . import extract_people
from . import recognize_people
from .visualize import visualize
from .util import temporary_directory

DETECTION_STEP = 5
RECOGNITION_STEP = DETECTION_STEP * 2


def gen_html(report_file, faces_df):
    men_pc, ages, frames_nums, attention_values = recognize_people.get_stats(faces_df)

    j2_env = Environment(
        loader=PackageLoader('boremeter', 'templates'),
        autoescape=select_autoescape(['html']),
    )

    template = j2_env.get_template('report.html')

    html_report = template.render(
        men_pc=men_pc,
        ages=str(ages.tolist()),
        time_arr=str(frames_nums.tolist()),
        attention_arr=str(attention_values.tolist()),
    )
    report_file.write(html_report)


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
    parser.add_argument('--gpu', action='store_true', help='switch for gpu computation')

    args = parser.parse_args()

    if args.frames_limit < 3:
        raise argparse.ArgumentTypeError('minimum frames_limit is 3')

    caffe_models_path = os.environ.get('CAFFE_MODELS_PATH') or args.caffe_models_path
    
    if args.gpu:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    # create temporary directory in the current directory where cropped faces will be stored
    with temporary_directory() as tmp_dir:

        print ('Extracting people')
        extracted_faces = extract_people.extract_faces(
            args.file.name,
            frames_limit=args.frames_limit,
            tmp_dir=tmp_dir,
            detection_step=DETECTION_STEP,
        )

        print ('Extracting statistics')
        recognized_faces_df = recognize_people.recognize_faces(
            detected_faces=extracted_faces,
            tmp_dir=tmp_dir,
            frames_limit=args.frames_limit,
            caffe_models_path=caffe_models_path,
            recognition_step=RECOGNITION_STEP,
        )

        print ('Generating html')
        gen_html(args.output_html, recognized_faces_df)

        if args.output_csv is not None:
            recognized_faces_df.to_csv(args.output_csv)

        if args.output_video is not None:
            print ('Visualizing')
            visualize(
                people_df=recognized_faces_df,
                input_videofile=args.file.name,
                output_videofile=args.output_video.name,
                frames_limit=args.frames_limit,
            )


if __name__ == '__main__':
    main()
