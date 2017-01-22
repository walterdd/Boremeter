from jinja2 import Environment, FileSystemLoader
import os

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
    gen_HTML('test.html', 0.8, [45,33,32,11,21,43,65,87,65,43,32,22,12], [1,2,3,4,5], [0.9, 0.6, 0.4,0.5, 0.4])

