#!/usr/bin/env python

"""
This is an implementation of a benchmark that was motivated by the model described in the paper:

T. Nowotny, R. Huerta, H. D. I. Abarbanel, and M. I. Rabinovich Self-organization in the olfactory system: One shot odor recognition in insects, Biol Cyber, 93 (6): 436-446 (2005), doi:10.1007/s00422-005-0019-7 

In contrast to the original model, this benchmark uses conductance based Hodgkin-Huxley type neurons and the feedforward inhibition through the lateral horn has been omitted.
"""
import random as py_random

from brian2 import *
import brian2genn
import sys

import benchmark_utils as bu

config = bu.prepare_benchmark(sys.argv)
bu.insert_general_benchmark_code(config)

# Number of neurons
N_PN = 100
N_iKC = int(2500*config['scale'])
N_eKC = 100
# Constants
g_Na = 7.15*uS
V_Na = 50*mV
g_Kd = 1.43*uS
V_Kd = -95*mV
g_L = 0.0267*uS
V_L = -63.56*mV
C_M = 0.3*nF
VT = -63*mV
# Those two constants are dummy constants, only used when populations only have
# either inhibitory or excitatory inputs
E_e = 0*mV
E_i = -92*mV
# Actual constants used for synapses
N_iKCeKC= N_iKC
if N_iKCeKC > 10000:
    N_iKCeKC = 10000
# scaling factor k for iKCeKC synaptic conductances
k = 2500/N_iKCeKC
if k < 1:
    k= 1
tau_PNiKC = 2*ms
tau_iKCeKC = 10*ms
tau_eKCeKC = 5*ms
w_eKCeKC = 75*nS
tau_pre = tau_post = 10*ms
dApre = 0.1*nS*k
dApost = -dApre
w_max = 3.75*nS*k

scale = .675

traub_miles = '''
dV/dt = -(1/C_M)*(g_Na*m**3*h*(V - V_Na) +
                g_Kd*n**4*(V - V_Kd) +
                g_L*(V - V_L) +
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
patterns = np.repeat(np.array([np.random.choice(N_PN, int(0.2*N_PN), replace=False) for _ in range(n_patterns)]), n_repeats, axis=0)
# Make variants of the patterns
to_replace = np.random.binomial(int(0.2*N_PN), p=p_perturb, size=n_patterns*n_repeats)
variants = []
for idx, variant in enumerate(patterns):
    np.random.shuffle(variant)
    if to_replace[idx] > 0:
        variant = variant[:-to_replace[idx]]
    new_indices = np.random.randint(N_PN, size=to_replace[idx])
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

PN = SpikeGeneratorGroup(N_PN, spike_indices, spike_times)

# iKC of the mushroom body
I_syn = '''g_PN_iKC : siemens
           I_syn = g_PN_iKC*(V - E_e): amp'''
eqs_iKC = Equations(traub_miles) + Equations(I_syn)
iKC = NeuronGroup(N_iKC, eqs_iKC, threshold='V>0*mV', refractory='V>0*mV',
                  method='exponential_euler')

# eKCs of the mushroom body lobe
I_syn = '''I_syn = g_iKC_eKC*(V - E_e) + g_eKC_eKC*(V - E_i): amp
           dg_iKC_eKC/dt = -g_iKC_eKC/tau_iKCeKC : siemens
           dg_eKC_eKC/dt = -g_eKC_eKC/tau_eKCeKC : siemens'''
eqs_eKC = Equations(traub_miles) + Equations(I_syn)
eKC = NeuronGroup(N_eKC, eqs_eKC, threshold='V>0*mV', refractory='V>0*mV',
                  method='exponential_euler')

# Synapses
PN_iKC = Synapses(PN, iKC, '''weight : siemens
                              ds/dt= -s/second :siemens
                              g_PN_iKC_post = s : siemens (summed)''',
                  on_pre='s += scale*weight')
iKC_eKC = Synapses(iKC, eKC,
                   '''w : siemens
                      dApre/dt = -Apre / tau_pre : siemens (event-driven)
                      dApost/dt = -Apost / tau_post : siemens (event-driven)''',
                   on_pre='''g_iKC_eKC += w
                             Apre += dApre
                             w = clip(w + Apost, 0, w_max)''',
                   on_post='''
                              Apost += dApost
                              w = clip(w + Apre, 0, w_max)''',
                   )
eKC_eKC = Synapses(eKC, eKC, on_pre='g_eKC_eKC += scale*w_eKCeKC')
bu.insert_benchmark_point()
PN_iKC.connect(p=0.15)

if (N_iKC > 10000):
    iKC_eKC.connect(p=float(10000)/N_iKC)
else:
    iKC_eKC.connect()
eKC_eKC.connect()
bu.insert_benchmark_point()

# First set all synapses as "inactive", then set 20% to active
PN_iKC.weight = '10*nS + 1.25*nS*randn()'
iKC_eKC.w = 'rand()*w_max/10*k'
iKC_eKC.w['rand() < 0.2'] = '(2.5*nS + 0.5*nS*randn())*k'
iKC.V = V_L
iKC.h = 1
iKC.m = 0
iKC.n = .5
eKC.V = V_L
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
    neurons = N_PN + N_iKC + N_eKC
    synapses = len(PN_iKC) + len(iKC_eKC) + len(eKC_eKC)
    bu.write_benchmark_results('Mbody_example', config, neurons, synapses, took)
else:
    for p, M in enumerate([PN_spikes, iKC_spikes, eKC_spikes]):
        subplot(2, 2, p+1)
        plot(M.t/ms, M.i, ',k')
        print('SpikeMon %d, average rate %.1f sp/s' %
              (p, M.num_spikes/(config['runtime']/second*len(M.source))))
    show()
