#!/bin/bash -e

for monitor in true false; do
    for threads in 0 -1; do
        for scaling in 0.05 0.1 0.25 0.5 1 2 4 8 16 32; do
            for repeat in 1 2 3; do
                echo Repeat $repeat: python $1 $scaling genn $threads 1 $monitor $2
                python $1 $scaling genn $threads 1 $monitor $2
                rm -r GeNNworkspace
	       done
       done
   done
done

# The really long runs (don't run with GeNN CPU-only, etc.)
if [ "$1" = "Mbody_example.py" ]; then
    for monitor in true false; do
        for scaling in 64 128 256 512; do
            for repeat in 1 2 3; do
                echo Repeat $repeat: python $1 $scaling genn 0 1 $monitor $2
                python $1 $scaling genn 0 1 $monitor $2
                rm -r GeNNworkspace
            done
        done
    done
fi
