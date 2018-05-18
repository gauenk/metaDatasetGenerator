#!/usr/bin/gnuplot --persist

FILENAME=ARG1
TO_TERMINAL=ARG2
print "FILENAME :",FILENAME

unset key

if (TO_TERMINAL eq "T" || TO_TERMINAL eq "True") set term dumb; else set output "histogram.ps"; set terminal postscript portrait color

plot FILENAME 
min_y = GPVAL_DATA_Y_MIN
max_y = GPVAL_DATA_Y_MAX

print "max_y ",max_y
print "min_y ",min_y

f(x) = mean_y
fit f(x) FILENAME u 1:1 via mean_y
stddev_y = sqrt(FIT_WSSR / (FIT_NDF + 1 ))

set label 1 gprintf("Mean = %g", mean_y) at 1000, min_y-20000 font "Arial,24"
set label 2 gprintf("Standard deviation = %g", stddev_y) at 1000, min_y-35000 font "Arial,24"

plot mean_y-stddev_y with filledcurves y1=mean_y lt 1 lc rgb "#bbbbdd",mean_y+stddev_y with filledcurves y1=mean_y lt 1 lc rgb "#bbddbb",mean_y w l lt 3 lw 10 lc rgb "#fa00000", FILENAME u 1 w p pt 2 lt 1 ps 1

