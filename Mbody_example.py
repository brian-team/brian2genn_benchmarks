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
import random as py_random

from brian2 import *
import brian2genn
import sys

import benchmark_utils as bu

config = bu.prepare_benchmark(sys.argv)
bu.insert_general_benchmark_code(config)

# Number of neurons
N_AL = 100
N_MB = int(2500*config['scale'])
N_LB = 100
# Constants
g_Na = 7.15*uS
E_Na = 50*mV
g_K = 1.43*uS
E_K = -95*mV
g_leak = 0.0267*uS
E_leak = -63.56*mV
C = 0.3*nF
VT = -63*mV
# Those two constants are dummy constants, only used when populations only have
# either inhibitory or excitatory inputs
E_e = 0*mV
E_i = -92*mV
# Actual constants used for synapses
NKCKC= N_MB
if NKCKC > 10000:
    NKCKC = 10000
g_scaling = NKCKC/2500
if g_scaling < 1:
    g_scaling= 1
tau_PN_LHI = 1*ms
tau_LHI_iKC = 3*ms
tau_PN_iKC = 2*ms
tau_iKC_eKC = 10*ms
tau_eKC_eKC = 5*ms
w_LHI_iKC = 8.75*nS
w_eKC_eKC = 75*nS
tau_pre = tau_post = 10*ms
dApre = 0.1*nS/g_scaling
dApost = -dApre
g_max = 3.75*nS/g_scaling

scale = .675

traub_miles = '''
dV/dt = -(1/C)*(g_Na*m**3*h*(V - E_Na) +
                g_K*n**4*(V - E_K) +
                g_leak*(V - E_leak) +
                I_syn) : volt
dm/dt = alpha_m*(1 - m) - beta_m*m : 1
dn/dt = alpha_n*(1 - n) - beta_n*n : 1
dh/dt = alpha_h*(1 - h) - beta_h*h : 1
alpha_m = 0.32*(mV**-1)*(13*mV-V+VT)/
         (exp((13*mV-V+VT)/(4*mV))-1.)/ms : Hz
beta_m = 0.28*(mV**-1)*(V-VT-40*mV)/
        (exp((V-VT-40*mV)/(5*mV))-1)/ms : Hz
alpha_h = 0.128*exp((17*mV-V+VT)/(18*mV))/ms : Hz
beta_h = 4./(1+exp((40*mV-V+VT)/(5*mV)))/ms : Hz
alpha_n = 0.032*(mV**-1)*(15*mV-V+VT)/
         (exp((15*mV-V+VT)/(5*mV))-1.)/ms : Hz
beta_n = .5*exp((10*mV-V+VT)/(40*mV))/ms : Hz
'''

# Principal neurons (Antennal Lobe)
n_patterns = 10
n_repeats = int(config['runtime']/second*10)
p_perturb = 0.1
patterns = np.repeat(np.array([np.random.choice(N_AL, int(0.2*N_AL), replace=False) for _ in range(n_patterns)]), n_repeats, axis=0)
# Make variants of the patterns
to_replace = np.random.binomial(int(0.2*N_AL), p=p_perturb, size=n_patterns*n_repeats)
variants = []
for idx, variant in enumerate(patterns):
    np.random.shuffle(variant)
    if to_replace[idx] > 0:
        variant = variant[:-to_replace[idx]]
    new_indices = np.random.randint(N_AL, size=to_replace[idx])
    variant = np.unique(np.concatenate([variant, new_indices]))
    variants.append(variant)

training_size = (n_repeats-10)
training_variants = []
for p in range(n_patterns):
    training_variants.extend(variants[n_repeats * p:n_repeats * p + training_size])
py_random.shuffle(training_variants)
sorted_variants = list(training_variants)
for p in range(n_patterns):
    sorted_variants.extend(variants[n_repeats * p + training_size:n_repeats * (p + 1)])

spike_times = np.arange(n_patterns*n_repeats)*50*ms + 1*ms + rand(n_patterns*n_repeats)*2*ms
spike_times = spike_times.repeat([len(p) for p in sorted_variants])
spike_indices = np.concatenate(sorted_variants)

PN = SpikeGeneratorGroup(N_AL, spike_indices, spike_times)

# iKC of the mushroom body
I_syn = '''I_syn = g_PN_iKC*(V - E_e): amp
           dg_PN_iKC/dt = -g_PN_iKC/tau_PN_iKC : siemens'''
eqs_iKC = Equations(traub_miles) + Equations(I_syn)
iKC = NeuronGroup(N_MB, eqs_iKC, threshold='V>0*mV', refractory='V>0*mV',
                  method='exponential_euler')


# eKCs of the mushroom body lobe
I_syn = '''I_syn = g_iKC_eKC*(V - E_e) + g_eKC_eKC*(V - E_i): amp
           dg_iKC_eKC/dt = -g_iKC_eKC/tau_iKC_eKC : siemens
           dg_eKC_eKC/dt = -g_eKC_eKC/tau_eKC_eKC : siemens'''
eqs_eKC = Equations(traub_miles) + Equations(I_syn)
eKC = NeuronGroup(N_LB, eqs_eKC, threshold='V>0*mV', refractory='V>0*mV',
                  method='exponential_euler')

# Synapses
PN_iKC = Synapses(PN, iKC, 'weight : siemens', on_pre='g_PN_iKC += scale*weight')
iKC_eKC = Synapses(iKC, eKC,
                   '''g_raw : siemens
                      dApre/dt = -Apre / tau_pre : siemens (event-driven)
                      dApost/dt = -Apost / tau_post : siemens (event-driven)
                      ''',
                   on_pre='''g_iKC_eKC += g_raw
                             Apre += dApre
                             g_raw = clip(g_raw + Apost, 0, g_max)
                             ''',
                   on_post='''
                              Apost += dApost
                              g_raw = clip(g_raw + Apre, 0, g_max)''',
                   )
eKC_eKC = Synapses(eKC, eKC, on_pre='g_eKC_eKC += scale*w_eKC_eKC')
bu.insert_benchmark_point()
PN_iKC.connect(p=0.15)

if (N_MB > 10000):
    iKC_eKC.connect(p=float(10000)/N_MB)
else:
    iKC_eKC.connect()
eKC_eKC.connect()
bu.insert_benchmark_point()

# First set all synapses as "inactive", then set 20% to active
PN_iKC.weight = '10*nS + 1.25*nS*randn()'
iKC_eKC.g_raw = 'rand()*g_max/10/g_scaling'
iKC_eKC.g_raw['rand() < 0.2'] = '(2.5*nS + 0.5*nS*randn())/g_scaling'
iKC.V = E_leak
iKC.h = 1
iKC.m = 0
iKC.n = .5
eKC.V = E_leak
eKC.h = 1
eKC.m = 0
eKC.n = .5

bu.insert_benchmark_point()
if config['monitor']:
    PN_spikes = SpikeMonitor(PN)
    iKC_spikes = SpikeMonitor(iKC)
    eKC_spikes = SpikeMonitor(eKC)

took = bu.do_and_measure_run(config)

if not config['debug']:
    neurons = N_AL + N_MB + N_LB
    synapses = len(PN_iKC) + len(iKC_eKC) + len(eKC_eKC)
    bu.write_benchmark_results('Mbody_example', config, neurons, synapses, took)
else:
    for p, M in enumerate([PN_spikes, iKC_spikes, eKC_spikes]):
        subplot(2, 2, p+1)
        plot(M.t/ms, M.i, ',k')
        print('SpikeMon %d, average rate %.1f sp/s' %
              (p, M.num_spikes/(config['runtime']/second*len(M.source))))
    show()
