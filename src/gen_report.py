from jinja2 import Environment, FileSystemLoader
import os
import sys
import extract_people as detect
import recognize_people as rec

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


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
    if len(sys.argv) < 2:
        print ("Provide video file path")
        sys.exit(1)
    video_file = sys.argv[1]
    output_file = sys.argv[2]
    print ("Extracting people.....")
    detect.fast_extract(video_file, visualize=True, frames_limit=100)
    print ("Extracting statistics.....")
    rec.recognize_people()
    print ("Generating html.....")
    stats = rec.get_stats()
    gen_HTML(output_file, stats[0], stats[1], stats[2], stats[3])


