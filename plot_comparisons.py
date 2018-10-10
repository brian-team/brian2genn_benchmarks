# coding=utf-8
from __future__ import unicode_literals

import os
import sys
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_benchmarks import load_benchmark, mean_and_std_fixed_time, FIGURE_EXTENSION


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
        cpu_results_single = benchmark.loc[(benchmark['device'] == 'cpp_standalone') & (benchmark['n_threads'] == 1)]

        for subset, name, ls in [(gpu_results, '{} – {}'.format(machine_name, GPU_name), '-'),
                                 (cpu_results, '{} – {} CPU cores'.format(machine_name, max_threads), ':'),
                                 (cpu_results_single, '{} – single thread'.format(machine_name), ':')]:
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
    ax.grid(b=True, which='major', color='#c0c0c0', linestyle='-',
            linewidth=0.5)
    ax.grid(b=True, which='minor', color='#c0c0c0', linestyle='-',
            linewidth=0.25)
    ax.set_xticks(np.log(sorted(used_n_neuron_values))[start::2])
    ax.set_xticklabels(sorted(used_n_neuron_values)[start::2], rotation=45)
    ax.set(xlabel='Model size (# neurons)',
           ylabel='Simulation time (s)',
           yscale='log',
           title=title)
    if legend:
        ax.legend(loc='upper left', frameon=True, edgecolor='none')


def plot_total_comparisons_only_GPU(benchmarks, reference_benchmarks, GPU_names,
                                    reference_labels, ax, title, legend=False,
                                    algorithm_details=False, select_benchmarks=None):
    colors = mpl.cm.tab10.colors
    if select_benchmarks is None:
        select_benchmarks = np.arange(len(benchmarks))
    ref_handles = []
    for idx, (reference_benchmark, reference_label) in enumerate(zip(reference_benchmarks, reference_labels)):
        c = (0.3*(idx+1), 0.3*(idx+1), 0.3*(idx+1))
        ref_handles.append(ax.plot(np.log(reference_benchmark['n_neurons'].values),
                                   reference_benchmark['duration_run_rel']['amin'],
                                   '-', label=reference_label, color=c)[0])
    algo_handles = []
    for idx, (benchmark, machine_name, GPU_name) in enumerate(zip(benchmarks,
                                                                  machine_names,
                                                                  GPU_names)):
        if idx not in select_benchmarks:
            continue
        # We do the log scale for the x axis manually -- easier to get the ticks/labels right
        # Only use Brian2GeNN GPU and maximum number of threads
        max_threads = benchmark.loc[benchmark['device'] == 'cpp_standalone']['n_threads'].max()
        gpu_results_pre = benchmark.loc[(benchmark['device'] == 'genn') &
                                        (benchmark['n_threads'] == 0) &
                                        (benchmark['algorithm'] == 'pre')]
        gpu_results_post = benchmark.loc[(benchmark['device'] == 'genn') &
                                         (benchmark['n_threads'] == 0) &
                                         (benchmark['algorithm'] == 'post')]

        name = GPU_name # '{} – {}'.format(machine_name, GPU_name)
        if algorithm_details:
            algo_handles.append(ax.plot(np.log(gpu_results_pre['n_neurons'].values),
                                        gpu_results_pre['duration_run_rel']['amin'],
                                        ':', color=colors[idx], label='“pre” strategy')[0])
            algo_handles.append(ax.plot(np.log(gpu_results_post['n_neurons'].values),
                                        gpu_results_post['duration_run_rel']['amin'],
                                        '--', color=colors[idx], label='“post” strategy')[0])
        if algorithm_details:
            style, label, mec, fc = '-', 'best strategy', colors[idx], colors[idx]
        else:
            style, label, mec, fc = 'o-', name, 'white', colors[idx]

        algo_handles.append(ax.plot(np.log(gpu_results_pre['n_neurons'].values),
                                    np.amin(np.vstack([gpu_results_pre['duration_run_rel']['amin'].values,
                                                       gpu_results_post['duration_run_rel']['amin'].values]),
                                            axis=0),
                                    style, label=label, color=fc, mec=mec)[0])
        used_n_neuron_values = benchmark['n_neurons'].unique()
    # Make sure we show the xtick label for the highest value
    if len(used_n_neuron_values) % 2 == 0:
        start = 1
    else:
        start = 0
    ax.grid(b=True, which='major', color='#e0e0e0', linestyle='-',
            linewidth=1.5)
    ax.grid(b=True, which='minor', color='#e0e0e0', linestyle='-',
            linewidth=0.5)
    ax.set_xticks(np.log(sorted(used_n_neuron_values))[start::2])
    ax.set_xticklabels(sorted(used_n_neuron_values)[start::2], rotation=45)
    ax.set(xlabel='Model size (# neurons)',
           ylabel='Simulation time (relative to biological time)',
           yscale='log',
           title=title)
    if algorithm_details:
        from matplotlib.lines import Line2D
        # Manually put a legend
        if legend:
            # legend for the CPUs/GPUs
            lines = ref_handles + [Line2D([0], [0], color=colors[idx], lw=2)
                                   for idx in select_benchmarks]
            labels = reference_labels + [GPU_names[idx] for idx in select_benchmarks]
            ax.legend(lines, labels, loc='upper left',
                      frameon=True, edgecolor='none')
        else:
            labels = (['', '', '']*(len(select_benchmarks) - 1) +
                      ['“pre” strategy', '“post”', 'best strategy'])
            ax.legend(algo_handles, labels, loc='upper left',
                      frameon=True, edgecolor='none', ncol=len(select_benchmarks),
                      columnspacing=0.)
    elif legend:
        ax.legend(loc='upper left', frameon=True, edgecolor='none')


def plot_necessary_runtime_across_gpus(benchmarks, reference_benchmark,
                                       labels, ax, title, legend=False):
    used_neuron_values = None
    for benchmark, label in zip(benchmarks, labels):
        benchmark = benchmark.loc[(benchmark['device'] == 'genn') &
                                  (benchmark['n_threads'] == 0)]

        # Merge the results from the two algorithms
        merged = pd.concat([benchmark['n_neurons'],
                            benchmark['total']['amin'],
                            benchmark['duration_run_rel']['amin']],
                           axis=1)
        merged.columns = ['n_neurons', 'total', 'duration_run_rel']
        grouped = merged.groupby(['n_neurons'])
        benchmark = grouped.agg([np.min]).reset_index()
        if used_neuron_values is None:
            used_neuron_values = benchmark['n_neurons'].values
        reference_benchmark = reference_benchmark.sort_values(by='n_neurons')
        # Only use those values where we have both kind of results
        available_sizes = set(benchmark['n_neurons'].unique()) & set(reference_benchmark['n_neurons'].unique())
        if len(set(benchmark['n_neurons'].unique()) - available_sizes):
            print('Benchmark {} has no results for sizes {} on the GPU'.format(title, set(benchmark['n_neurons'].unique()) - available_sizes))
        if len(set(reference_benchmark['n_neurons'].unique()) - available_sizes):
            print('Benchmark {} has no results for sizes {} on the CPU'.format(
                title, set(reference_benchmark['n_neurons'].unique()) - available_sizes))
        benchmark = benchmark.loc[benchmark['n_neurons'].isin(available_sizes)]
        reference_benchmark = reference_benchmark.loc[reference_benchmark['n_neurons'].isin(available_sizes)]

        variable_time_gpu = benchmark['duration_run_rel']['amin'].values
        fixed_time_gpu = benchmark['total']['amin'].values - variable_time_gpu
        variable_time_cpu = reference_benchmark['duration_run_rel']['amin'].values
        fixed_time_cpu = reference_benchmark['total']['amin'].values - variable_time_cpu
        # Check assumptions
        necessary = (fixed_time_cpu - fixed_time_gpu)/(variable_time_gpu - variable_time_cpu)
        # If GPU takes longer per simulated second, no way to get a faster sim
        necessary[variable_time_gpu > variable_time_cpu] = np.NaN
        # Fixed time is already lower for GPU
        necessary[fixed_time_gpu < fixed_time_cpu] = 0
        ax.plot(np.log(benchmark['n_neurons']).unique(), necessary, 'o-',
                mec='white', label=label)

    # Make sure we show the xtick label for the highest value
    if len(used_neuron_values) % 2 == 0:
        start = 1
    else:
        start = 0

    ax.set_xticklabels(sorted(used_neuron_values)[start::2], rotation=45)
    ax.grid(b=True, which='major', color='#e0e0e0', linestyle='-',
            linewidth=1.5)
    ax.grid(b=True, which='minor', color='#e0e0e0', linestyle='-',
            linewidth=0.5)
    ax.set(xticks=np.log(used_neuron_values)[start::2],
           xlabel='Model size (# neurons)',
           ylabel='necessary biological runtime (s)',
           yscale='log', title=title)
    if legend:
        ax.legend(loc='lower left', frameon=True, edgecolor='none')


if __name__ == '__main__':
    plt.style.use('figures.conf')
    target_dir = 'benchmark_results/comparisons'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    benchmark_dirs = ['benchmark_results/2018-10-02_f152b85d2726',
                      'benchmark_results/2018-10-05_vuvuzela',
                      'benchmark_results/2018-10-04_jwc09n012']

    float_dtypes_per_benchmark = [('float32', 'float64'),
                                  ('float32', ),
                                  ('float64', )]
    reference_dir = 'benchmark_results/2018-10-02_f152b85d2726'
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

    monitor_str = '_no_monitor'
    fig, axes = plt.subplots(2, 2, sharey='row', sharex='row',
                             figsize=(6.33, 6.33*1.33))
    fig_gpu, axes_gpu = plt.subplots(2, 2, sharey='row', sharex='row',
                                     figsize=(6.33, 6.33*1.33))

    for col, float_dtype in enumerate(['float64', 'float32']):
        precision = 'single precision' if float_dtype == 'float32' else 'double precision'
        precision_short = 'single' if float_dtype == 'float32' else 'double'
        for ax, title, fname in [(axes[1, col], 'COBAHH', 'benchmarks_COBAHH.txt'),
                                 (axes[0, col], 'Mbody', 'benchmarks_Mbody_example.txt')]:
            benchmarks = [mean_and_std_fixed_time(load_benchmark(dirname, fname),
                                                  monitor=False, float_dtype=float_dtype)
                          for dirname in benchmark_dirs]

            plot_total_comparisons(benchmarks, machine_names, gpu_names,
                                   ax, title + ' – ' + precision,
                                   legend=(ax == axes[0, 1]))

        for ax, title, fname in [(axes_gpu[1, col], 'COBAHH', 'benchmarks_COBAHH.txt'),
                                 (axes_gpu[0, col], 'Mbody', 'benchmarks_Mbody_example.txt')]:
            benchmarks = [mean_and_std_fixed_time(load_benchmark(dirname, fname),
                                                  monitor=False, float_dtype=float_dtype)
                          for dirname in benchmark_dirs]

            reference = mean_and_std_fixed_time(load_benchmark(reference_dir, fname),
                                                monitor=False,
                                                float_dtype=float_dtype)
            reference24 = reference.loc[(reference['device'] == 'cpp_standalone') &
                                       (reference['n_threads'] == 24)]
            reference1 = reference.loc[(reference['device'] == 'cpp_standalone') &
                                       (reference['n_threads'] == 1)]
            plot_total_comparisons_only_GPU(benchmarks, [reference1,
                                                         reference24],
                                            gpu_names,
                                            ['CPU / 1 thread',
                                             'CPU / 24 threads'],
                                            ax, title + ' – ' + precision,
                                            legend=(ax == axes_gpu[1, 0]))

    fig.tight_layout()
    fig.savefig(os.path.join(target_dir,
                             'runtime_comparison_{}'.format(FIGURE_EXTENSION)))
    plt.close(fig)

    fig_gpu.tight_layout()
    fig_gpu.savefig(os.path.join(target_dir,
                             'gpu_runtime_comparison{}'.format(FIGURE_EXTENSION)))
    plt.close(fig_gpu)

    fig_gpu_algos, axes_gpu_algos = plt.subplots(1, 2, sharey='row',
                                                 figsize=(6.33, 6.33*0.666))
    float_dtype = 'float32'
    for ax_detail, title, fname in [(axes_gpu_algos[1],
                                     'COBAHH', 'benchmarks_COBAHH.txt'),
                                    (axes_gpu_algos[0],
                                     'Mbody', 'benchmarks_Mbody_example.txt')]:
        benchmarks = [mean_and_std_fixed_time(load_benchmark(dirname, fname),
                                              monitor=False,
                                              float_dtype=float_dtype)
                      for dirname in benchmark_dirs]
        reference = mean_and_std_fixed_time(
            load_benchmark(reference_dir, fname),
            monitor=False,
            float_dtype=float_dtype)
        reference24 = reference.loc[(reference['device'] == 'cpp_standalone') &
                                    (reference['n_threads'] == 24)]
        reference1 = reference.loc[(reference['device'] == 'cpp_standalone') &
                                   (reference['n_threads'] == 1)]
        plot_total_comparisons_only_GPU(benchmarks, [reference1,
                                                     reference24],
                                        gpu_names,
                                        ['CPU / 1 thread',
                                         'CPU / 24 threads'],
                                        ax_detail, title + ' – ' + precision,
                                        legend=(ax_detail == axes_gpu_algos[0]),
                                        algorithm_details=True,
                                        select_benchmarks=[0, 2])

    fig_gpu_algos.tight_layout()
    fig_gpu_algos.savefig(os.path.join(target_dir,
                                       'gpu_runtime_comparison_algos{}'.format(FIGURE_EXTENSION)))
    plt.close(fig_gpu_algos)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey='row',
                                            figsize=(6.33, 6.33))
    for ax, title, fname in [(ax_right, 'COBAHH', 'benchmarks_COBAHH.txt'),
                             (ax_left, 'Mbody', 'benchmarks_Mbody_example.txt')]:
        benchmarks = [mean_and_std_fixed_time(load_benchmark(dirname, fname),
                                              monitor=False, float_dtype=float_dtype)
                      for dirname, float_dtypes in zip(benchmark_dirs, float_dtypes_per_benchmark)
                      for float_dtype in float_dtypes]

        reference = mean_and_std_fixed_time(load_benchmark(reference_dir, fname),
                                            monitor=False,
                                            float_dtype='float64')
        reference = reference.loc[(reference['device'] == 'cpp_standalone') &
                                  (reference['n_threads'] == 24)]
        labels = ['%s (%s)' % (gpu_name, 'single' if float_dtype == 'float32' else 'double')
                  for gpu_name, float_dtypes in zip(gpu_names, float_dtypes_per_benchmark)
                  for float_dtype in float_dtypes]
        plot_necessary_runtime_across_gpus(benchmarks, reference, labels,
                                           ax, legend=(ax == ax_left),
                                           title=title)

    fig.tight_layout()
    fname = os.path.join(target_dir, 'necessary_biological_runtime_across_GPUs{}{}'.format(monitor_str,
                                                                                         FIGURE_EXTENSION))
    fig.savefig(fname)
    plt.close(fig)
