#!/bin/sh
python Mbody_example.py 8 cpp_standalone 0 10 store float32 validation
python Mbody_example.py 8 cpp_standalone 0 10 store float64 validation
python Mbody_example.py 8 cpp_standalone 24 10 store float32 validation
python Mbody_example.py 8 cpp_standalone 24 10 store float64 validation
python COBAHH.py 8 cpp_standalone 0 10 store float32 validation
python COBAHH.py 8 cpp_standalone 0 10 store float64 validation
python COBAHH.py 8 cpp_standalone 24 10 store float32 validation
python COBAHH.py 8 cpp_standalone 24 10 store float64 validation
