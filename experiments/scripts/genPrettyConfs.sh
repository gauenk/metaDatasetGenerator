#!/bin/bash
# Usage:
# ./experiments/scripts/genPrettyConfs.sh SIZE 

SIZE=$1

if [ "$SIZE" == "" ]
then
    echo "Usage: ./experiments/scripts/genPrettyConfs.sh SIZE"
    exit 1
fi

for REPEAT in `seq 0 9`
do
    ./tools/prettyConfMat.py --size $SIZE --repeat $REPEAT
done
