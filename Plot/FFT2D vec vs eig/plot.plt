#!/usr/bin/gnuplot -persist

set terminal pngcairo
set output "./plot.png"

set title "FFT2D on square matrix"
set xlabel "Matrix width"
set ylabel "μs"
set grid
plot "out.dat" u 1:2:xtic(1) w l lw 2 lc 1 title "Sequential vec","out.dat" u 1:3:xtic(1) w l lw 2 lc 2 title "Parallel OMP vec", "out.dat" u 1:4:xtic(1) w l dt 3 lw 2 lc 1 title "Sequential eig","out.dat" u 1:5:xtic(1) w l dt 3 lw 2 lc 2 title "Parallel OMP eig"
