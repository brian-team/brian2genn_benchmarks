#!/bin/bash
cd /root/brian2genn/scripts/benchmarking
echo "Hello" >file.txt
cp -f *.txt /root/parent
bash run_benchmarks_genn.sh COBAHH.py
cp -f *.txt /root/parent
bash run_benchmarks_genn.sh Mbody_example.py
cp -f *.txt /root/parent
bash run_benchmarks.sh COBAHH.py
cp -f *.txt /root/parent
bash run_benchmarks.sh Mbody_example.py
cp -f *.txt /root/parent
