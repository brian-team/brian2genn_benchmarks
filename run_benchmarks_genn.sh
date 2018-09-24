#!/bin/bash -e

if [ "$3" == "test" ]; then
    SCALING="1 8"
    SCALING_BIG="32"
    MONITORS="false"
    N_REPEATS=1
else
    SCALING=""
    SCALING_BIG="64 128 256 512"
    if [ "$1" = "COBAHH.py" ]; then
        MONITORS="true false"
        SCALING_BIG="64 128"
    else
        MONITORS="false"
        SCALING_BIG="0.05 0.1 0.25 0.5 1 2 4 8 16 32 64 128 256 512 1024 2048 4096"
    fi
    N_REPEATS=3
fi

for monitor in $MONITORS; do
    for threads in 0 -1; do
        for scaling in $SCALING; do
            for float_dtype in float32 float64; do
                for repeat in $(seq $N_REPEATS); do
                    echo Repeat $repeat: python $1 $scaling genn $threads 1 $monitor $float_dtype $2
                    python $1 $scaling genn $threads 1 $monitor $float_dtype $2
                    rm -r GeNNworkspace
                done
	        done
       done
   done
done

# The really long runs (don't run with GeNN CPU-only, etc.)
for monitor in $MONITORS; do
    for scaling in $SCALING_BIG; do
        for float_dtype in float32 float64; do
            for repeat in $(seq $N_REPEATS); do
                echo Repeat $repeat: python $1 $scaling genn 0 1 $monitor $float_dtype $2
                python $1 $scaling genn 0 10 $monitor $float_dtype $2
                rm -r GeNNworkspace
            done
        done
    done
done
