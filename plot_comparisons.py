# coding=utf-8
from __future__ import unicode_literals

import os
import sys
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


from plot_benchmarks import load_benchmark, mean_and_std_fixed_time


def plot_total_comparisons(benchmarks, machine_names, GPU_names, ax, title, legend=False):
    colors = mpl.cm.tab10.colors
    for idx, (benchmark, machine_name, GPU_name) in enumerate(zip(benchmarks,
                                                                  machine_names,
                                                                  GPU_names)):
        # We do the log scale for the x axis manually -- easier to get the ticks/labels right
        # Only use Brian2GeNN GPU and maximum number of threads
        max_threads = benchmark.loc[benchmark['device'] == 'cpp_standalone']['n_threads'].max()
        gpu_results = benchmark.loc[(benchmark['device'] == 'genn') & (benchmark['n_threads'] == 0)]
        cpu_results = benchmark.loc[(benchmark['device'] == 'cpp_standalone') & (benchmark['n_threads'] == max_threads)]

        for subset, name, ls in [(gpu_results, '{} – {}'.format(machine_name, GPU_name), '-'),
                                 (cpu_results, '{} – {} CPU cores'.format(machine_name, max_threads), ':')]:
            if len(subset) == 0:
                continue
            ax.plot(np.log(subset['n_neurons'].values),
                    subset['duration_run']['amin'],
                    'o-', label=name, color=colors[idx], linestyle=ls,
                    mec='white')
        used_n_neuron_values = benchmark['n_neurons'].unique()
        # Make sure we show the xtick label for the highest value
    if len(used_n_neuron_values) % 2 == 0:
        start = 1
    else:
        start = 0
    ax.grid(b=True, which='minor', color='#b0b0b0', linestyle='-',
            linewidth=0.33)
    ax.set_xticks(np.log(sorted(used_n_neuron_values))[start::2])
    ax.set_xticklabels(sorted(used_n_neuron_values)[start::2], rotation=45)
    ax.set(xlabel='Model size (# neurons)',
           ylabel='Simulation time (s)',
           yscale='log',
           title=title)
    if legend:
        ax.legend(loc='upper left', frameon=True, edgecolor='none')


def plot_total_comparisons_only_GPU(benchmarks, reference_benchmark, GPU_names,
                                    reference_label, ax, title, legend=False):
    colors = mpl.cm.tab10.colors
    ax.plot(np.log(reference_benchmark['n_neurons'].values),
            reference_benchmark['duration_run_rel']['amin'],
            '-', label=reference_label, color='black')

    for idx, (benchmark, machine_name, GPU_name) in enumerate(zip(benchmarks,
                                                                  machine_names,
                                                                  GPU_names)):
        # We do the log scale for the x axis manually -- easier to get the ticks/labels right
        # Only use Brian2GeNN GPU and maximum number of threads
        max_threads = benchmark.loc[benchmark['device'] == 'cpp_standalone']['n_threads'].max()
        gpu_results = benchmark.loc[(benchmark['device'] == 'genn') & (benchmark['n_threads'] == 0)]

        name = '{} – {}'.format(machine_name, GPU_name)
        ax.plot(np.log(gpu_results['n_neurons'].values),
                gpu_results['duration_run_rel']['amin'],
                'o-', label=name, color=colors[idx], mec='white')
        used_n_neuron_values = benchmark['n_neurons'].unique()
        # Make sure we show the xtick label for the highest value
    if len(used_n_neuron_values) % 2 == 0:
        start = 1
    else:
        start = 0
    ax.grid(b=True, which='minor', color='#b0b0b0', linestyle='-',
            linewidth=0.33)
    ax.set_xticks(np.log(sorted(used_n_neuron_values))[start::2])
    ax.set_xticklabels(sorted(used_n_neuron_values)[start::2], rotation=45)
    ax.set(xlabel='Model size (# neurons)',
           ylabel='Simulation time (relative to biological time)',
           yscale='log',
           title=title)
    if legend:
        ax.legend(loc='upper left', frameon=True, edgecolor='none')


if __name__ == '__main__':
    plt.style.use('figures.conf')
    target_dir = 'benchmark_results/comparisons'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    benchmark_dirs = ['benchmark_results/2018-09-13_c6a4ae8d7e8b',  # Used as reference for CPU
                      'benchmark_results/2018-09-13_inf900777/',
                      'benchmark_results/2018-09-13_vuvuzela',
                      'benchmark_results/2018-09-14_jwc09n009/']
    machine_names = []
    gpu_names = []

    for dirname in benchmark_dirs:
        try:
            machine_name = open(os.path.join(dirname, 'machine_name.txt')).read().strip()
        except (IOError, OSError):
            warnings.warn('Could not open {} to get a human-readable '
                          'machine name.'.format(os.path.join(dirname, 'machine_name.txt')))
            machine_name = os.path.abspath(dirname).split(os.sep)[-1][12:]
        try:
            gpu_name = open(os.path.join(dirname, 'gpu_name.txt')).read().strip()
        except (IOError, OSError):
            warnings.warn('Could not open {} to get a human-readable '
                          'GPU name.'.format(os.path.join(dirname, 'gpu_name.txt')))
            gpu_name = 'GPU'
        machine_names.append(machine_name)
        gpu_names.append(gpu_name)

    for monitor in [True, False]:
        monitor_str = '' if monitor else '_no_monitor'
        fig, axes = plt.subplots(2, 2, sharey='row', sharex='row',
                                 figsize=(6.33, 6.33*1.33))
        fig_gpu, axes_gpu = plt.subplots(2, 2, sharey='row', sharex='row',
                                         figsize=(6.33, 6.33*1.33))
        for col, float_dtype in enumerate(['float64', 'float32']):
            precision = 'single precision' if float_dtype == 'float32' else 'double precision'
            for ax, title, fname in [(axes[1, col], 'COBAHH', 'benchmarks_COBAHH.txt'),
                                     (axes[0, col], 'Mbody', 'benchmarks_Mbody_example.txt')]:
                benchmarks = [mean_and_std_fixed_time(load_benchmark(dirname, fname),
                                                      monitor=monitor, float_dtype=float_dtype)
                              for dirname in benchmark_dirs]

                plot_total_comparisons(benchmarks, machine_names, gpu_names,
                                       ax, title + ' – ' + precision,
                                       legend=(ax == axes[0, 1]))

            for ax, title, fname in [(axes_gpu[1, col], 'COBAHH', 'benchmarks_COBAHH.txt'),
                                     (axes_gpu[0, col], 'Mbody', 'benchmarks_Mbody_example.txt')]:
                benchmarks = [mean_and_std_fixed_time(load_benchmark(dirname, fname),
                                                      monitor=monitor,
                                                      float_dtype=float_dtype)
                              for dirname in benchmark_dirs]
                reference = mean_and_std_fixed_time(load_benchmark(benchmark_dirs[0], fname),
                                                    monitor=monitor,
                                                    float_dtype=float_dtype)
                reference = reference.loc[(reference['device'] == 'cpp_standalone') &
                                          (reference['n_threads'] == 24)]
                plot_total_comparisons_only_GPU(benchmarks, reference,
                                                gpu_names, 'Core i9-7920X@2.9GHz – 24 threads',
                                                ax, title + ' – ' + precision,
                                                legend=(ax == axes_gpu[0, 1]))
        fig.tight_layout()
        fig.savefig(os.path.join(target_dir,
                                 'runtime_comparison{}.png'.format(monitor_str)))
        plt.close(fig)

        fig_gpu.tight_layout()
        fig_gpu.savefig(os.path.join(target_dir,
                                 'gpu_runtime_comparison{}.png'.format(monitor_str)))
        plt.close(fig_gpu)