#! /bin/bash

LABEL=$(date -I)_"$HOSTNAME"_PROFILING
echo Using label: $LABEL

RESULT_DIR="benchmark_results/$LABEL"
mkdir -p $RESULT_DIR

RUNTIME=1

for TRIAL in 1 2 3; do
  echo Trial $TRIAL

  # Mbody example (10,240,200 neurons --> scale: 4096)
  SCALE=4096
  python Mbody_example.py $SCALE genn 0 $RUNTIME false float64 $LABEL pre true | tee -a $RESULT_DIR/Mbody_genn_double.log
  python Mbody_example.py $SCALE genn 0 $RUNTIME false float32 $LABEL pre true | tee -a $RESULT_DIR/Mbody_genn_single.log
  python Mbody_example.py $SCALE cpp_standalone 24 $RUNTIME false float64 $LABEL true | tee -a $RESULT_DIR/Mbody_cpp_double.log
  python Mbody_example.py $SCALE cpp_standalone 24 $RUNTIME false float32 $LABEL true | tee -a $RESULT_DIR/Mbody_cpp_single.log

  # COBAHH (512,000 neurons --> scale 128)
  SCALE=128
  python COBAHH.py $SCALE genn 0 $RUNTIME false float64 $LABEL pre true | tee -a $RESULT_DIR/COBAHH_genn_double.log
  python COBAHH.py $SCALE genn 0 $RUNTIME false float32 $LABEL pre true | tee -a $RESULT_DIR/COBAHH_genn_single.log
  python COBAHH.py $SCALE cpp_standalone 24 $RUNTIME false float64 $LABEL true | tee -a $RESULT_DIR/COBAHH_cpp_double.log
  python COBAHH.py $SCALE cpp_standalone 24 $RUNTIME false float32 $LABEL true | tee -a $RESULT_DIR/COBAHH_cpp_single.log

done