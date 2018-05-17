#!/bin/bash
# Usage:
# ./experiments/scripts/verifyDatasetLoad.sh GPU DATASET IMAGE_SET

set -e

START_IDX=$1
END_IDX=$2
REPEAT=$3

if [ "$START_IDX" == "" ] || [ "$END_IDX" == "" ] || [ "$REPEAT" == "" ]
then
    echo "Usage: ./experiments/scripts/genMixData.sh START_IDX END_IDX REPEAT"
    echo "example: ./experiments/scripts/verifyDatasetLoad.sh 4 4 3"
    exit 1
fi

time ./tools/generateMixtureDatasets.py --dataset_range_start $START_IDX --dataset_range_end $END_IDX --repeat $REPEAT --cfg ./experiments/cfgs/generateMixtureDatasets.yml
