#! /bin/bash

# Define a basic label to keep the benchmark results organized
LABEL="$(date -I)_$HOSTNAME"
echo Using label: $LABEL

bash run_benchmarks_genn.sh Mbody_example.py $LABEL
bash run_benchmarks_genn.sh COBAHH.py $LABEL
bash run_benchmarks_cpp.sh Mbody_example.py $LABEL
bash run_benchmarks_cpp.sh COBAHH.py $LABEL
