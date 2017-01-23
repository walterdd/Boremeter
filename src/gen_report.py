from jinja2 import Environment, FileSystemLoader
import os
import sys
import extract_people as detect
import recognize_people as rec
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument

add_arg('file',                 type=str,                           help='Limit of frames to process.')
add_arg('--frames_limit',       default=200, type=int,              help='Limit of frames to process.')
add_arg('--output_html',        default='out.html', type=str,       help='Report file.')
add_arg('--output_video',       default='NAN', type=str,            help='Videofile to visualise bbs on video.')
args = parser.parse_args()

def array_to_string(arr):
	return '[' + ', '.join(str(x) for x in arr) + ']'


def gen_HTML(filename, men_pc, ages, time_arr, attention_arr):
    j2_env = Environment(loader=FileSystemLoader(THIS_DIR), trim_blocks=True)
    template = j2_env.get_template('template.html')

    with open(filename, "wb") as fh:
        fh.write(template.render(men_pc=men_pc, 
            ages=array_to_string(ages), 
            time_arr=array_to_string(time_arr), 
            attention_arr=array_to_string(attention_arr)))


if __name__ == "__main__":
    if not args.file:
        print "Provide input!"
        sys.exit()

    print ("Extracting people.....")

    if (args.output_video != 'NAN'):
        detect.fast_extract(args.file, visualize=True, frames_limit=200, output_file_name=args.output_video)
    else:
        detect.fast_extract(args.file, visualize=False, frames_limit=200)

    print ("Extracting statistics.....")
    rec.recognize_people()
    print ("Generating html.....")
    stats = rec.get_stats()
    gen_HTML(args.output_html, stats[0], stats[1], stats[2], stats[3])


