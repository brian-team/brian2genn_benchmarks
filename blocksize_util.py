import os
from brian2 import *

fname= 'GeNNworkspace/blocksizes'
folder= 'benchmark_results/'+sys.argv[3]
name= sys.argv[1]
with open('%s/blocksizes_%s.txt' % (folder, name), 'a') as f:
    with open(fname, 'r') as bf:
        wline= sys.argv[2]
        for line in bf:
            line= line.strip()
            items= line.split(':')
            wline= wline + ' ' + items[-1]
        wline= wline + '\n'
        f.write(wline)
            
