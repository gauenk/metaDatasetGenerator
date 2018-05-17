#!/bin/bash
# Usage:
# ./experiments/scripts/verifyDatasetLoad.sh GPU DATASET IMAGE_SET

set -e

DATASET=$1
IMAGE_SET=$2
NET_lc=${NET,,}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

if [ "$DATASET" == "" ] || [ "$IMAGE_SET" == "" ]
then
    echo "Usage: ./experiments/scripts/verifyDatasetLoad.sh DATASET IMAGESET"
    echo "example: ./experiments/scripts/verifyDatasetLoad.sh pascal_voc_2007 train"
    exit 1
fi

time ./tools/imdbReport.py \
  --imdb ${DATASET}"-"${IMAGE_SET}"-default" \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml
