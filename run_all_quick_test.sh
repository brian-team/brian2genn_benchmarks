#! /bin/bash

# Define a basic label to keep the benchmark results organized
LABEL="$(date -I)_$HOSTNAME"_test
echo Using label: $LABEL

RESULT_DIR="benchmark_results/$LABEL"
mkdir -p $RESULT_DIR
EXAMPLES="Mbody_example COBAHH"

for EXAMPLE in $EXAMPLES; do
    bash run_benchmarks_genn.sh "$EXAMPLE".py $LABEL test 2>&1 | tee $RESULT_DIR/"$EXAMPLE"_genn.log
    bash run_benchmarks_cpp.sh "$EXAMPLE".py $LABEL test 2>&1 | tee $RESULT_DIR/"$EXAMPLE"_cpp.log
done

python plot_benchmarks.py $RESULT_DIR