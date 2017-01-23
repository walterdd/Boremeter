## Auditory Tracking

*Open source app for tracking auditory attention on video*

---

## The Idea

Given a video this app extracts audience socio-demographic statistics and tracks viewers boredom during lectures or presentations.

<img src="https://github.com/walterdd/Auditory_tracking/blob/master/dogg.jpg" width="224">

[slides](https://docs.google.com/presentation/d/14mCydv-_sYkVHxImUnIX6PWRfpsfL49311rG099QPvc/edit#slide=id.g19ead2f26b_0_16)

[People tracking example video](https://www.youtube.com/watch?v=LFJhAiqAA3c)


## Setup

### With Docker

We strongly suggest you to run AuditoryTracking in [Docker](https://www.docker.com). That will make life easier. 

To install AuditoryTracking just clone the repository and build with Docker

```bash
$ git clone https://github.com/walterdd/Auditory_tracking.git
$ cd Auditory_tracking
$ docker build -t atracking .
```

## Usage
To run AuditoryTracking in Docker use

```bash
$ atracker.sh -id=$INPUT_DIR -if=$INPUT_FILENAME -ov=$OUTPUT_FILENAME -oh=$OUTPUT_HTML -fl=100

--input-video|-iv  -  just the name of input video file
--input-dir|-id  -  full path to a directory with input video
--output_video|-ov  -  name of an output video file with visualisation
--output_html|-oh  -  name of an output filename with report
--frames_limit|-fl  -  number of frames to process
```

### Without Docker
### Requirements

You can give it a try!

+ Ubuntu 14.04 or older
+ Python 2.7
+ Caffe
+ OpenCV3
+ Python requirements listed in requirements.txt

Download pre-trained caffe nets and save them to Auditory_tracking/caffe_models:

[dex_imdb_wiki.caffemodel](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_imdb_wiki.caffemodel)

[gender.caffemodel](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.caffemodel)

Run

```bash
$ python gen_report.py --file=$INPUT_FILENAME --output_video=$OUTPUT_FILENAME \
                       --output_html=$OUTPUT_HTML --frames_limit=100
```
