#!/bin/bash

# Max size
maxSize=128

# Initial size
val=4

# Number of iterations to time to get the average.
iterations=25

# Create or truncate the output .dat file
echo -e "Row size\tSequential timing\tParallel timing (us)" > ./out.dat

# Loop through iterations
for ((val; val<=$maxSize; val<<=1)); do
    # Run the command and save the output
    output=$(./test2D.exe -N $val -iTT $iterations | grep "on average" | awk '{print $(NF-1)}')
    
    # Append data to the .dat file
    echo -e "$val\t${output//$'\n'/ }" >> out.dat
done

# Plot data
gnuplot plot.plt