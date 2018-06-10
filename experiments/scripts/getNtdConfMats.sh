#!/bin/bash
# Usage:
# ./experiments/scripts/getNtdConfMats.sh SIZE 

SIZE=$1

if [ "$SIZE" == "" ]
then
    echo "Usage: ./experiments/scripts/getNtdConfMats.sh SIZE"
fi
for REPEAT in `seq 0 9`
do
    ./tools/ntdConfMats.py --size $SIZE --repeat $REPEAT
done
