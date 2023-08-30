#!/bin/bash

# BEFORE RUNNING THIS SCRIPT COMPILE WITH ONLY PARALLEL IMPLEMENTATION TRUE

exePath="../../FFT2D_Vec/test2D.exe"

# Initial size
val=4
# Max size
maxSize=128

# Number of iterations to time to get the average.
iterations=50

# IF YOU CHANGE minThreads,maxThreads CHANGE ALSO IN PLOT.PLT
# Initial threads
minThreads=2
# Max threads
maxThreads=12

# Check if the executable file exists
if [[ ! -f "$exePath" ]]; then
    echo "Executable file not found: $exePath"
    exit 1
fi

# Create or truncate the output .dat file
echo -e "Row size\tSequential timing\tParallel timing (us)" > ./out.dat

# Time every matrix size
for ((val; val<=$maxSize; val<<=1)); do

    # For every matrix size try different threads number:
    output=""
    for ((threads=minThreads; threads<=$maxThreads; threads+=2)); do
        output="$output$($exePath -N $val -iTT $iterations -nTH $threads | grep -m 1 "on average" | awk '{print $(NF-1)}')\t"
    done

    # Append data to the .dat file:
    # size  timeWithMinThreads  ..  ..  timeWithMaxThreads
    echo -e "$val\t$output" >> out.dat
done

# Plot data
gnuplot plot.plt