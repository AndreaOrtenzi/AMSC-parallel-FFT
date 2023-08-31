#!/usr/bin/gnuplot -persist

set terminal pngcairo
set output "./plot.png"

set title "FFT2D num threads"
set xlabel "Matrix width"
set ylabel "μs"
set grid

# Define the thread counts and line styles
thread_counts = "2 4 6 8 10 12"
line_styles = "l lp l lp l lp"

# Set up the plot command
plot_command = "plot"
do for [i=1:words(thread_counts)] {
    thread_count = word(thread_counts, i)
    line_style = word(line_styles, i)
    plot_command = plot_command . \
        sprintf("'out.dat' u 1:%d:xtic(1) w %s lw 2 title '%s threads', ", \
                i+1, line_style, thread_count)
}

# Remove trailing comma and execute the plot command
plot_command = substr(plot_command, 1, strlen(plot_command)-2)
eval(plot_command)
