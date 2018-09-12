#!/bin/bash
sudo docker run --runtime=nvidia -v `pwd`/:/root/brian2genn_benchmarks/ -it brian2genn_benchmarks /bin/bash /root/brian2genn_benchmarks/run-benchmarks-in-docker.sh
