## Boremeter

*Open source app for tracking auditory boredom on video*

---

## The Idea

Given a video this app extracts audience socio-demographic statistics and tracks viewers boredom during lectures or presentations.

<img src="https://github.com/walterdd/Auditory_tracking/blob/master/dogg.jpg" width="224">

[slides](https://docs.google.com/presentation/d/14mCydv-_sYkVHxImUnIX6PWRfpsfL49311rG099QPvc/edit#slide=id.g19ead2f26b_0_16)

[People tracking example video](https://www.youtube.com/watch?v=LFJhAiqAA3c)


## Setup

### With Docker

We strongly suggest you to run Boremeter in [Docker](https://www.docker.com). That will make life easier. 

To install Boremeter just clone the repository and build with Docker

```bash
$ git clone https://github.com/walterdd/Boremeter.git
$ cd Boremeter
$ docker build -t boremeter .
```

## Usage
To run Boremeter in Docker use

```bash
$ docker run -v {host directory path}:{container directory path} -it boremeter
$ boremeter --file={input video file}
```

command line flags:

```bash
--file  -  input video file
--output_video -  path to output .avi file with visualisation
--output_html  -  path to output .html file with report
--frames_limit  -  number of frames to process
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

### Setup

```bash
$ git clone https://github.com/walterdd/Boremeter.git
$ cd Boremeter
$ python setup.py install
```

