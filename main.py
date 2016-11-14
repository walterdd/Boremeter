import extract_people as detect
import recognize_people as rec
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print ("Provide video file path")
        sys.exit(1)
    video_file = sys.argv[1]
    detect.extract_people(video_file, visualize=True, frames_limit=100)
    rec.recognize_people()
