#!/usr/bin/gnuplot -persist

set terminal png
set output "./plot.png"

set title "FFT2D on square matrix using vector<vector>"
set xlabel "Matrix width"
set ylabel "us"
set grid
plot "out.dat" u 1:2:xtic(1) w l lw 2 title "Sequential","out.dat" u 1:3:xtic(1) w l lw 2 title "Parallel OMP"