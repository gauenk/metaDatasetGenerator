#!/bin/bash
1;4205;0c# Usage:
# ./experiments/scripts/verifyDatasetLoad.sh GPU DATASET IMAGE_SET

set -e

GPU_ID=$1
DATASET=$2
IMAGE_SET=$3
NET_lc=${NET,,}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

if [ "$GPU_ID" == "" ] || [ "$DATASET" == "" ] || [ "$IMAGE_SET" == "" ]
then
    echo "Usage: ./experiments/scripts/verifyDatasetLoad.sh GPUID DATASET IMAGESET"
    echo "example: ./experiments/scripts/verifyDatasetLoad.sh 0 caltech train"
    exit 1
fi

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${DATASET}/VGG16/faster_rcnn_end2end/solver.prototxt \
  --imdb ${DATASET}"-"${IMAGE_SET}"-default" \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml
