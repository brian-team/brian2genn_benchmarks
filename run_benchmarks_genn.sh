#!/bin/bash -e

if [ "$3" == "test" ]; then
    SCALING="1 8"
    MONITORS="false"
    RUNTIME=1
    N_REPEATS=1
else
    SCALING="0.05 0.1 0.25 0.5 1 2 4 8 16 32"
    MONITORS="false"
    RUNTIME=10
    N_REPEATS=3
    if [ "$1" = "COBAHH.py" ]; then
        if [ "$3" == "short" ]; then
            SCALING="0.05 0.1 0.25 0.5 1 2 4 8 16 32 64 128"
        else
            SCALING="0.05 0.1 0.25 0.5 1 2 4 8 16 32 64 128 256 512"
        fi
    else
        if [ "$3" == "short" ]; then
            SCALING="0.05 0.1 0.25 0.5 1 2 4 8 16 32 64 128 256 512 1024 2048 4096"
        else
            SCALING="0.05 0.1 0.25 0.5 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192"
        fi
    fi
fi

for monitor in $MONITORS; do
    for para in pre post; do
        for scaling in $SCALING; do
            for float_dtype in float32 float64; do
                for repeat in $(seq $N_REPEATS); do
                    echo Repeat $repeat: python $1 $scaling genn 0 1 $monitor $float_dtype $2 $para 1
                    python $1 $scaling genn 0 $RUNTIME $monitor $float_dtype $2 $para 1
                    rm -r GeNNworkspace
                done
            done
        done
    done
done
