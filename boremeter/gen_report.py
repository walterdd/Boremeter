from jinja2 import Environment, PackageLoader, select_autoescape
import os
import shutil
import sys
import extract_people as detect
import recognize_people as rec
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--file',               type=str,                               help='input video file')
parser.add_argument('--frames_limit',       default=200, type=int,                  help='limit of frames to process')
parser.add_argument('--output_html',        default='out.html', type=str,           help='path to output .html file with report')
parser.add_argument('--output_video',       default='NAN', type=str,                help='path to output .avi file with' \
                                                                                         'visualisation of bounding boxes')
parser.add_argument('--output_csv',         default='recognized.csv', type=str,     help='path to output table with information' \
                                                                                         'about all detected faces')
parser.add_argument('--caffe_models_path',  default='/root/caffe/models', type=str, help='path to directory with pre-trained models' \
                                                                                         'must contain "age.prototxt", "gender.prototxt",' \
                                                                                         ' "age.caffemodel", "gender.caffemodel"')
args = parser.parse_args()

def array_to_string(arr):
	return '[' + ', '.join(str(x) for x in arr) + ']'


def gen_HTML(filename, men_pc, ages, time_arr, attention_arr):
    j2_env = Environment(
        loader=PackageLoader('boremeter', 'templates'),
        autoescape=select_autoescape(['html'])
    )
    template = j2_env.get_template('report.html')

    with open(filename, "wb") as fh:
        fh.write(template.render(men_pc=men_pc, 
            ages=array_to_string(ages), 
            time_arr=array_to_string(time_arr), 
            attention_arr=array_to_string(attention_arr)))


def validate_input_path(path):
    if not os.path.isfile(args.file):
        print "Input path %s does not exist" % path
        sys.exit()


def validate_output_path(path):
    if not os.path.dirname(path)=="" and not os.path.isdir(os.path.dirname(path)):
        print "Output directory %s does not exist" % os.path.dirname(path)
        sys.exit()


def setup(args):
    if not args.file:
        print "Provide input --file"
        sys.exit()
    if args.frames_limit < 2:
        print "Frames limit --frames_limit should be larger"

    validate_input_path(args.file)
    validate_output_path(args.output_html)
    validate_output_path(args.output_video)
    validate_output_path(args.output_csv)

    # create temporary directory in the current directory where cropped faces will be stored
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
    os.mkdir("tmp")

    # set environment variable, paths to caffe models
    os.environ["caffe_models"] = args.caffe_models_path


def farewell():
    shutil.rmtree("tmp")


def main():
    setup(args)

    print ("Extracting people.....")

    if (args.output_video != 'NAN'):
        detect.fast_extract(args.file, visualize=True, frames_limit=args.frames_limit, output_file_name=args.output_video)
    else:
        detect.fast_extract(args.file, visualize=False, frames_limit=args.frames_limit)

    print ("Extracting statistics.....")
    rec.recognize_people()
    print ("Generating html.....")
    stats = rec.get_stats(table=args.output_csv)
    gen_HTML(args.output_html, stats[0], stats[1], stats[2], stats[3])

    farewell()


if __name__ == "__main__":
    main()


