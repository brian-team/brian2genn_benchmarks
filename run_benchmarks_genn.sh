#!/bin/bash -e

if [ "$3" == "test" ]; then
    SCALING="1 8"
    SCALING_BIG="32"
    MONITORS="false"
    N_REPEATS=1
    BLOCKSIZE=0
else
    if [ "$3" == "blocksize" ]; then
	SCALING=""
	SCALING_BIG="0.05 0.1 0.25 0.5 1 2 4 8 16 32 64 128 256 512"
	MONITORS="false"
	N_REPEATS=1
	BLOCKSIZE=1
    else
	SCALING="0.05 0.1 0.25 0.5 1 2 4 8 16 32"
	if [ "$1" = "COBAHH.py" ]; then
            MONITORS="true false"
            SCALING_BIG="64 128"
	else
            MONITORS="false"
            SCALING=""
	    SCALING_BIG="0.05 0.1 0.25 0.5 1 2 4 8 16 32 64 128 256 512 1024 2048 4096"
	fi
	N_REPEATS=3
	BLOCKSIZE=0
    fi
fi

for monitor in $MONITORS; do
    for para in pre post; do
	for threads in 0 -1; do
            for scaling in $SCALING; do
		for float_dtype in float32 float64; do
                    for repeat in $(seq $N_REPEATS); do
			    echo Repeat $repeat: python $1 $scaling genn $threads 1 $monitor $float_dtype $2 $para 1
			    python $1 $scaling genn $threads 1 $monitor $float_dtype $2 $para 1
			    rm -r GeNNworkspace
			done
		    done
		done
	    done
       done		
done

# The really long runs (don't run with GeNN CPU-only, etc.)
for monitor in $MONITORS; do
    for para in pre post; do
	for scaling in $SCALING_BIG; do
            for float_dtype in float32 float64; do
		for repeat in $(seq $N_REPEATS); do
			echo Repeat $repeat: python $1 $scaling genn 0 1 $monitor $float_dtype $2 $para 1
			python $1 $scaling genn 0 1 $monitor $float_dtype $2  $para 1
			if [ "$BLOCKSIZE" == "1" ]; then
			    cd GeNNworkspace
			    echo "$GENN_PATH/lib/bin/genn-buildmodel.sh magicnetwork_model.cpp &> buildmodel.log"
			    $GENN_PATH/lib/bin/genn-buildmodel.sh magicnetwork_model.cpp &> buildmodel.log
			    cat buildmodel.log | grep "block size:" | tee > blocksizes
			    cd ..
			    python blocksize_util.py ${1%.py} $scaling $2
			fi
			rm -r GeNNworkspace
		done
	    done
	done
    done	    
done
