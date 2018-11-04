#!/bin/bash

FILE=$1
TO_TERMINAL=$2

if [ "$FILE" == "" ] || [ "$TO_TERMINAL" == "" ]
then
    echo "Usage: ./experiments/scripts/plotAlReport.sh FILE_TO_PLOT PLOT_TO_TERMINAL_BOOL?(T|F)"
    exit 1
fi

gnuplot -c ./experiments/scripts/plotAlReport.gp "$FILE" "$TO_TERMINAL"
