#!/bin/bash
module load python_gpu/3.7.4

# Store the current date and time
printf -v DATETIME '%(%Y-%m-%d_%H:%M:%S)T' -1

# Submit the program
bsub -n 8 -W 24:00 -N -B -R "rusage[mem=16384,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" -o output_$DATETIME.txt 'python PartC_AutoSyllClust.py' 

