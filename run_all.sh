#! /bin/bash

# Define a basic label to keep the benchmark results organized
LABEL="$(date -I)_$HOSTNAME"
echo Using label: $LABEL

RESULT_DIR="benchmark_results/$LABEL"
mkdir -p $RESULT_DIR
nvidia-smi -L > $RESULT_DIR/gpu.txt
lscpu > $RESULT_DIR/cpuinfo.txt || echo "could not get machine info with lscpu"
EXAMPLES="Mbody_example COBAHH"

for EXAMPLE in $EXAMPLES; do
    echo Running examples with Brian2GeNN
    bash run_benchmarks_genn.sh "$EXAMPLE".py $LABEL $1 2>&1 | tee -a $RESULT_DIR/"$EXAMPLE"_genn.log
    echo Running examples with C++ standalone mode
    bash run_benchmarks_cpp.sh "$EXAMPLE".py $LABEL $1 2>&1 | tee -a $RESULT_DIR/"$EXAMPLE"_cpp.log
done

python plot_benchmarks.py $RESULT_DIR
