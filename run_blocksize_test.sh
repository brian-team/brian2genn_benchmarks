#! /bin/bash

# Define a basic label to keep the benchmark results organized
LABEL="$(date -I)_$HOSTNAME"_blocksize
echo Using label: $LABEL

RESULT_DIR="benchmark_results/$LABEL"
mkdir -p $RESULT_DIR
EXAMPLES="COBAHH"

for EXAMPLE in $EXAMPLES; do
    bash run_benchmarks_genn.sh "$EXAMPLE".py $LABEL blocksize 2>&1 | tee $RESULT_DIR/"$EXAMPLE"_genn.log
done

