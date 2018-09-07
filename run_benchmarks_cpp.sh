#!/bin/bash -e

for monitor in true false; do
    for threads in 1 2 4 8 16; do
        for scaling in 0.05 0.1 0.25 0.5 1 2 4 8 16 32; do
            for repeat in 1 2 3; do
                echo Repeat $repeat: python $1 $scaling cpp_standalone $threads 1 $monitor $2
                python $1 $scaling cpp_standalone $threads 1 $monitor $2
                rm -r output
	       done
       done
   done
done

# The really long runs (don't run with low # of threads, etc.)
if [ "$1" = "Mbody_example.py" ]; then
    for monitor in true false; do
        for threads in 8 16; do
            for scaling in 64 128 256 512; do
                for repeat in 1 2 3; do
                    echo Repeat $repeat: python $1 $scaling cpp_standalone $threads 1 $monitor $2
                    python $1 $scaling cpp_standalone $threads 1 $monitor $2
                    rm -r output
                done
            done
        done
    done
fi
