#!/bin/bash

# BEFORE RUNNING THIS SCRIPT COMPILE WITH SEQUENTIAL AND PARALLEL IMPLEMENTATIONs TRUE

vecVecExePath="../../FFT2D_Vec/test2D.exe"
eigenExePath="../../FFT2D/test2D.exe"

# Max size
maxSize=64

# Initial size
val=4

# Number of iterations to time to get the average.
iterations=25

# Create or truncate the output .dat file
echo -e "Row size\tSequential timing\tParallel timing (us)" > ./out.dat

# Check if the executable file exists
if [[ -n "$vecVecExePath" && ! -f "$vecVecExePath" ]]; then
    echo "Executable file not found: $vecVecExePath"
    exit 1
fi

# Check if the executable file exists
if [[ -n "$eigenExePath" && ! -f "$eigenExePath" ]]; then
    echo "Executable file not found: $vecVecExePath"
    exit 1
fi

# Loop through iterations
for ((val; val<=$maxSize; val<<=1)); do
    # Run the command and save the output
    if [[ -n "$vecVecExePath" ]]; then
    vecOutput=$($vecVecExePath -N $val -iTT $iterations | grep "on average" | awk '{print $(NF-1)}')
    else
    vecOutput="0\t0"
    fi
    if [[ -n "$eigenExePath" ]]; then
    eiOutput=$($eigenExePath -N $val -iTT $iterations | grep "on average" | awk '{print $(NF-1)}')
    else
    eiOutput="0\t0"
    fi
    # Append data to the .dat file:
    # size  seqTimeVec  parTimeVec  seqTimeEig  parTimeEig
    echo -e "$val\t${vecOutput//$'\n'/$'\t'}\t${eiOutput//$'\n'/$'\t'}" >> out.dat
done

# Plot data
gnuplot plot.plt