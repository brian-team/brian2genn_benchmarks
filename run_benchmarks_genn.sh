#!/bin/bash -e

if [ "$3" == "test" ]; then
    SCALING="1 8"
    RUNTIME=1
    N_REPEATS=1
else
    RUNTIME=10
    N_REPEATS=3
    if [ "$1" = "COBAHH.py" ]; then
    SCALING="0.05 0.1 0.25 0.5 1 2 4 8 16 32"
        if [ "$3" == "short" ]; then
            SCALING_BIG="64 128"
        else
            SCALING_BIG="64 128 256 512"
        fi
    else
        SCALING="0.05 0.1 0.25 0.5 1 2 4 8 16 32 64 128 256 512 1024"
        if [ "$3" == "short" ]; then
            SCALING_BIG="2048 4096"
        else
            SCALING_BIG="2048 4096 8192"
        fi
    fi
fi

MONITORS="false"
kernel_timing=false

for monitor in $MONITORS; do
    for para in pre post; do
        for scaling in $SCALING; do
            for float_dtype in float32 float64; do
                for repeat in $(seq $N_REPEATS); do
                    echo Repeat $repeat: python $1 $scaling genn 0 $RUNTIME $monitor $float_dtype $2 $para $kernel_timing
                    python $1 $scaling genn 0 $RUNTIME $monitor $float_dtype $2 $para $kernel_timing
                    rm -r GeNNworkspace
                done
            done
        done
    done
done

RUNTIME=1
for monitor in $MONITORS; do
    for para in pre post; do
        for scaling in $SCALING_BIG; do
            for float_dtype in float32 float64; do
                for repeat in $(seq $N_REPEATS); do
                    echo Repeat $repeat: python $1 $scaling genn 0 $RUNTIME $monitor $float_dtype $2 $para $kernel_timing
                    python $1 $scaling genn 0 $RUNTIME $monitor $float_dtype $2 $para $kernel_timing
                    rm -r GeNNworkspace
                done
            done
        done
    done
done