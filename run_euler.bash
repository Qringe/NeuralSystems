#!/bin/bash
module load python/3.6.0
module load python/3.7.1

# Store the current date and time
printf -v DATETIME '%(%Y-%m-%d_%H:%M:%S)T' -1

# Submit the program
bsub -W 24:00 -n 8 -R rusage[mem=16000,scratch=1000] -N -B -o output_$DATETIME.txt python PartC_AutoSyllClust.py 

