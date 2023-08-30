#!/usr/bin/gnuplot -persist

set terminal pngcairo
set output "./plot.png"

set title "FFT2D num threads"
set xlabel "Matrix width"
set ylabel "us"
set grid
plot "out.dat" u 1:2:xtic(1) w l lw 2 title "2 threads","out.dat" u 1:3:xtic(1) w l lw 2 title "4 threads", "out.dat" u 1:4:xtic(1) w l lw 2 title "6 threads","out.dat" u 1:5:xtic(1) w l lw 2 title "8 threads", "out.dat" u 1:5:xtic(1) w l lw 2 title "10 threads", "out.dat" u 1:5:xtic(1) w l lw 2 title "12 threads"
