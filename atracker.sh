#!/bin/bash

# As long as there is at least one more argument, keep looping
while [[ $# -gt 0 ]]; do
    key="$1"
    case "$key" in
        # This is an arg=value type option. Will catch -id=value or --input-dir=value
        -id=*|--input-dir=*)
        INPUT_DIR="${key#*=}";
        echo $INPUT_DIR
        ;;
        # This is an arg=value type option. Will catch -if=value or --input-file=value
        -iv=*|--input-video*)
        INPUT_FILE="${key#*=}";
        echo $INPUT_FILE
        ;;
        # This is an arg=value type option. Will catch -od=value or --output-dir=value
        -fl=*|--frames_limit=*)
        # No need to shift here since the value is part of the same string
        FRAMES_LIMIT="${key#*=}";
        echo $FRAMES_LIMIT
        ;;
        # This is an arg=value type option. Will catch -o=value or --output-file=value
        -oh=*|--output_html=*)
        # No need to shift here since the value is part of the same string
        OUTPUT_HTML="${key#*=}";
        echo $OUTPUT_HTML
        ;;
        # This is an arg=value type option. Will catch -o=value or --output-file=value
        -ov=*|--output_video=*)
        # No need to shift here since the value is part of the same string
        OUTPUT_VIDEO="${key#*=}";
        echo $OUTPUT_VIDEO
        ;;
        *)
        # Do whatever you want with extra options
        echo "Unknown option '$key'"
        exit 1
        ;;
    esac
    # Shift after checking all the cases to get the next option
    shift
done
if [[ -z $OUTPUT_VIDEO ]] ; then
    echo 'specify --output_video|-ov'
    exit 1
fi
if [[ -z $OUTPUT_HTML ]] ; then
    echo 'specify --output_html|-oh'
    exit 1
fi
if [[ -z $INPUT_FILE ]] ; then
    echo 'specify --input-file|-if'
    exit 1
fi
if [[ -z $INPUT_DIR ]] ; then
    echo 'specify --input-dir|-id'
    exit 1
fi
if [[ -z $FRAMES_LIMIT ]] ; then
    echo 'specify --frames_limit|-fl'
    exit 1
fi

# Workdirectory directories in docker container.
# Host workdir with inputs will be mounted into it
CONTAINER_WORKDIR=/root/workdir

docker run -v $INPUT_DIR:$CONTAINER_WORKDIR -it atracking python gen_report.py --file=$CONTAINER_WORKDIR/$INPUT_FILE \
																			   --output_html=$CONTAINER_WORKDIR/$OUTPUT_HTML \
																			   --output_video=$CONTAINER_WORKDIR/OUTPUT_VIDEO \
																			   --frames_limit=$FRAMES_LIMIT


