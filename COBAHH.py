#!/usr/bin/env python

"""
This is an implementation of a benchmark described
in the following review paper:

Simulation of networks of spiking neurons: A review of tools and strategies (2006).
Brette, Rudolph, Carnevale, Hines, Beeman, Bower, Diesmann, Goodman, Harris, Zirpe,
Natschlaeger, Pecevski, Ermentrout, Djurfeldt, Lansner, Rochel, Vibert, Alvarez, Muller,
Davison, El Boustani and Destexhe.
Journal of Computational Neuroscience

Benchmark 3: random network of HH neurons with exponential synaptic conductances

Clock-driven implementation
(no spike time interpolation)

R. Brette - Dec 2007
"""
import os
import sys

from brian2 import *
import brian2genn

import benchmark_utils as bu

config = bu.prepare_benchmark(sys.argv)
bu.insert_general_benchmark_code(config)

# Parameters
area = 20000 * umetre ** 2
C_M = (1 * ufarad * cm ** -2) * area
g_L = (5e-5 * siemens * cm ** -2) * area

V_L = -60 * mV
V_Kd = -90 * mV
V_Na = 50 * mV
g_Na = (100 * msiemens * cm ** -2) * area
g_Kd = (30 * msiemens * cm ** -2) * area
VT = -63 * mV
# Time constants
tau_E = 5 * ms
tau_I = 10 * ms
# Reversal potentials
V_E = 0 * mV
V_I = -80 * mV

# The model
eqs = Equations('''
dV/dt = (g_L*(V_L-V)+g_E*(V_E-V)+g_I*(V_I-V)-
         g_Na*(m*m*m)*h*(V-V_Na)-
         g_Kd*(n*n*n*n)*(V-V_Kd))/C_M : volt
dm/dt = alpha_m*(1-m)-beta_m*m : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dg_E/dt = -g_E*(1./tau_E) : siemens
dg_I/dt = -g_I*(1./tau_I) : siemens
alpha_m = 0.32*(mV**-1)*(13*mV-V+VT)/
         (exp((13*mV-V+VT)/(4*mV))-1.)/ms : Hz
beta_m = 0.28*(mV**-1)*(V-VT-40*mV)/
        (exp((V-VT-40*mV)/(5*mV))-1)/ms : Hz
alpha_h = 0.128*exp((17*mV-V+VT)/(18*mV))/ms : Hz
beta_h = 4./(1+exp((40*mV-V+VT)/(5*mV)))/ms : Hz
alpha_n = 0.032*(mV**-1)*(15*mV-V+VT)/
         (exp((15*mV-V+VT)/(5*mV))-1.)/ms : Hz
beta_n = .5*exp((10*mV-V+VT)/(40*mV))/ms : Hz
''')

P = NeuronGroup(int(4000 * config['scale']), model=eqs,
                threshold='V>-20*mV', refractory=3 * ms,
                method='exponential_euler')
Pe = P[:int(3200 * config['scale'])]
Pi = P[int(3200 * config['scale']):]
Ce = Synapses(Pe, P, 'w_E : siemens (constant)', on_pre='g_E+=w_E')
Ci = Synapses(Pi, P, 'w_I : siemens (constant)', on_pre='g_I+=w_I')
bu.insert_benchmark_point()
Ce.connect(p=1000./len(P))
Ci.connect(p=1000./len(P))
bu.insert_benchmark_point()
# Initialization
P.V = 'V_L + (randn() * 5 - 5)*mV'
P.g_E = '(randn() * 1.5 + 4) * 10.*nS'
P.g_I = '(randn() * 12 + 20) * 10.*nS'
Ce.w_E = 'rand() * 1e-9*nS'
Ci.w_I = 'rand() * 1e-9*nS'
bu.insert_benchmark_point()

# Record 1% of all neurons
if config['monitor']:
    trace = StateMonitor(P, 'V', record=np.arange(0, P.N, P.N // 100))
    spikemon = SpikeMonitor(P)

if config['debug']:
    popratemon = PopulationRateMonitor(P)

took = bu.do_and_measure_run(config)

if not config['debug']:
    neurons = len(P)
    synapses = len(Ce) + len(Ci)
    bu.write_benchmark_results('COBAHH', config, neurons, synapses, took)
    if config.get('monitor_store', False):
        folder = os.path.join('benchmark_results', config['label'])
        device_str = 'cpp' if config['device'] == 'cpp_standalone' else 'genn'
        float_str = 'single' if config['float_dtype'] == 'float32' else 'double'
        if config['device'] == 'cpp_standalone':
            thread_str = '_{}_threads'.format(config['threads'])
        else:
            thread_str = ''
        fname = os.path.join(folder,
                             'COBAHH_spikes_{}_{}{}.npz'.format(device_str,
                                                                float_str,
                                                                thread_str))
        np.savez_compressed(fname,
                            config=config,
                            spike_times=spikemon.t_[:],
                            spike_indices=spikemon.i[:])
else:
    print('Number of spikes: %d' % spikemon.num_spikes)
    print('Mean firing rate: %.1f Hz' %
          (spikemon.num_spikes/(config['runtime']*len(P))))
    subplot(311)
    plot(spikemon.t/ms, spikemon.i, ',k')
    subplot(312)
    plot(trace.t/ms, trace.V[:].T[:, :5])
    subplot(313)
    plot(popratemon.t/ms, popratemon.smooth_rate(width=1*ms))
    show()
