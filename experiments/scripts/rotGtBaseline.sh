#!/bin/bash
# Usage:
# ./experiments/scripts/rotGtBaseline.sh

echo "This script generates a baseline accuracury for rotating an image."

outputFile="./output/faster_rcnn/rotGtBaseline.txt"
rm $outputFile

set -x
set -e

export PYTHONUNBUFFERED="True"

DATASET=$1

case $DATASET in
    imagenet)
	TRAIN_IMDB="imagenet-train"
	#TEST_IMDB="imagenet_test"
	#TEST_IMDB="imagenet_very_short_train"
	TEST_IMDB="imagenet-val1"
	PT_DIR="imagenet"
	ITERS=100000
	;;
    pascal_voc)
	TRAIN_IMDB="pascal_voc-trainval"
	#TEST_IMDB="pascal_voc-test"
	TEST_IMDB="pascal_voc-vshort"
	PT_DIR="pascal_voc"
	ITERS=120000
	;;
    pascal_voc_2007)
	TRAIN_IMDB="pascal_voc_2007-trainval"
	TEST_IMDB="pascal_voc_2007-test"
	PT_DIR="pascal_voc"
	ITERS=70000
	;;
    pascal_voc_2012)
	TRAIN_IMDB="pascal_voc_2012-trainval"
	TEST_IMDB="pascal_voc_2012-val"
	PT_DIR="pascal_voc"
	ITERS=70000
	;;
    coco)
	# This is a very long and slow training schedule
	# You can probably use fewer iterations and reduce the
	# time to the LR drop (set in the solver to 350,000 iterations).
	TRAIN_IMDB="coco-train"
	TEST_IMDB="coco-testdev2015"
	PT_DIR="coco"
	ITERS=490000
	;;
    cam2)
	# this is cam2 data :-)
	TRAIN_IMDB="cam2-trainval"
	#TEST_IMDB="cam2_2017_test" #"cam2_2017_trainval"
	TEST_IMDB="cam2-all" #"cam2_2017_trainval"
	PT_DIR="cam2"
	ITERS=10000
	;;
    sun)
	TRAIN_IMDB="sun-train"
	#TEST_IMDB="sun_2012_taste"
	TEST_IMDB="sun-test"
	PT_DIR="sun"
	ITERS=10000
	;;
    caltech)
	TRAIN_IMDB="caltech-train"
	TEST_IMDB="caltech-test"
	PT_DIR="caltech"
	ITERS=10000
	;;
    kitti)
	TRAIN_IMDB="kitti-train"
	TEST_IMDB="kitti-val"
	#TEST_IMDB="kitti_2013_train"
	PT_DIR="kitti"
	ITERS=70000
	;;
    inria)
	TRAIN_IMDB="inria-train"
	TEST_IMDB="inria-test"
	PT_DIR="inria"
	#ITERS=150000
	ITERS=200000
	;;
    *)
	echo "No dataset given"
	exit
	;;
esac

ROTATION=(0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90)

#ROTATION=(0 15 30 45 60 75 90)
for i in "${ROTATION[@]}"
do
    echo "TESTING FOR ROTATION @ "$i
    ./tools/reval.py --imdb "$TEST_IMDB"-default ./output/faster_rcnn_end2end/pascal_voc/pascal_voc_2012_iter_70000/ --gt --rot $i

done


for i in "${ROTATION[@]}"
do
    fn=($(ls | grep "results_faster-rcnn_${i}_${PT_DIR}*"))
    value=$(head -n2 $fn | tail -n1 | cut -d':' -f2 | cut -d'	' -f2 | sed -e 's/\s*//g')
    echo $value $i >> $outputFile
done
