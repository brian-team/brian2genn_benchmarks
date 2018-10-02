#!/bin/bash -e

N_CPUS=$([ $(uname) == 'Darwin' ] && sysctl -n hw.physicalcpu_max ||
         lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)
N_CPUS_LOGICAL=$([ $(uname) == 'Darwin' ] && sysctl -n hw.logicalcpu_max ||
                 lscpu -p | egrep -v '^#' | wc -l)
echo "Physical number of processors: $N_CPUS"
echo "Logical number of processors: $N_CPUS_LOGICAL"

if [ "$3" == "test" ]; then
    THREADS="1 $N_CPUS"
    THREADS_BIG="$N_CPUS"
    SCALING="1 8"
    SCALING_BIG="32"
    MONITORS="false"
    N_REPEATS=1
else
    if [ $N_CPUS -ne $N_CPUS_LOGICAL ]; then
        THREADS="1 $N_CPUS $N_CPUS_LOGICAL"
        THREADS_BIG="$N_CPUS $N_CPUS_LOGICAL"
    else
        THREADS="1 $N_CPUS"
        THREADS_BIG="$N_CPUS"
    fi
    SCALING="0.05 0.1 0.25 0.5 1 2 4 8 16 32"
    MONITORS="false"
    if [ "$1" == "COBAHH.py" ]; then
        if [ "$3" == "short" ]; then
            SCALING_BIG="64 128"
        else
            SCALING_BIG="64 128 256 512"
        fi
    else
        if [ "$3" == "short" ]; then
            SCALING_BIG="64 128 256 512 1024 2048 4096"
        else
            SCALING_BIG="64 128 256 512 1024 2048 4096 8192"
        fi
    fi
    N_REPEATS=3
fi

RUNTIME=10
for monitor in $MONITORS; do
    for threads in $THREADS; do
        for scaling in $SCALING; do
            for float_dtype in float32 float64; do
                for repeat in $(seq $N_REPEATS); do
                    echo Repeat $repeat: python $1 $scaling cpp_standalone $threads $RUNTIME $monitor $float_dtype $2
                    python $1 $scaling cpp_standalone $threads $RUNTIME $monitor $float_dtype $2
                    rm -r output
                done
            done
       done
   done
done

# The really long runs (don't run with low # of threads, etc.)
RUNTIME=1
for monitor in $MONITORS; do
    for threads in $THREADS_BIG; do
        for scaling in $SCALING_BIG; do
            for float_dtype in float32 float64; do
                for repeat in $(seq $N_REPEATS); do
                    echo Repeat $repeat: python $1 $scaling cpp_standalone $threads $RUNTIME $monitor $float_dtype $2
                    python $1 $scaling cpp_standalone $threads $RUNTIME $monitor $float_dtype $2
                    rm -r output
                done
            done
        done
    done
done