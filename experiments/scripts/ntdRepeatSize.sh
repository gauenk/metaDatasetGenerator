#!/bin/bash

SIZE=$1

if [ "$SIZE" == "" ]
then
    echo "Usage: ./experiments/scripts/ntdRepeatSize.sh SIZE"
    exit 1
fi

echo "Running NTD with size $SIZE"
./tools/ntdConfMats.py --size 1000 --repeat 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --cfg ./experiments/cfgs/name_that_dataset.yml
