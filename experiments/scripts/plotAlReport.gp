#!/usr/bin/gnuplot --persist

FILENAME=ARG1
TO_TERMINAL=ARG2
print "FILENAME :",FILENAME

unset key
set datafile separator ','

if (TO_TERMINAL eq "T" || TO_TERMINAL eq "True") set term dumb; else set output "alPlot.ps"; set terminal postscript portrait color


set xlabel 'Entropy' font 'Courier,24'
set ylabel '% Error Reduction' font 'Courier,24'
plot [0:1][-.001:.001] FILENAME u 12:13
