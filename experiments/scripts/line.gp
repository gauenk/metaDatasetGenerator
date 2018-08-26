#!/usr/bin/gnuplot --persist

FILENAME=ARG1
TO_TERMINAL=ARG2
print "FILENAME :",FILENAME

unset key

if (TO_TERMINAL eq "T" || TO_TERMINAL eq "True") set term dumb; else set output "line.ps"; set terminal postscript portrait color

set xlabel "Degrees of Rotation"
set ylabel "Model Accuracy (AP Person)"

plot FILENAME u 2:1
