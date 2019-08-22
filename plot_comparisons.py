# coding=utf-8
from __future__ import unicode_literals
from __future__ import print_function

import os
import sys
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import xticks

from plot_benchmarks import (load_benchmark, mean_and_std_fixed_time,
                             FIGURE_EXTENSION, MBody_xticks, COBAHH_xticks)


def plot_total_comparisons(benchmarks, machine_names, GPU_names, ax, title,
                           ticks, legend=False, colors=None):
    if colors is None:
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
        # Make sure we show the xtick label for the highest value
    ax.grid(b=True, which='major', color='#c0c0c0', linestyle='-',
            linewidth=0.5)
    ax.grid(b=True, which='minor', color='#c0c0c0', linestyle='-',
            linewidth=0.25)
    ax.set_xticks(np.log(ticks))
    ax.set_xticklabels(ticks, rotation=45)
    ax.set(xlabel='Number of neurons',
           ylabel='Simulation time (s)',
           yscale='log',
           title=title)
    if legend:
        ax.legend(loc='upper left', frameon=True, edgecolor='none')


def plot_total_comparisons_only_GPU(benchmarks, reference_benchmarks, GPU_names,
                                    reference_labels, ax, title, ticks, legend=False,
                                    algorithm_details=False, select_benchmarks=None,
                                    colors=None):
    if colors is None:
        colors = mpl.cm.tab10.colors
    if select_benchmarks is None:
        select_benchmarks = np.arange(len(benchmarks))
    ref_handles = []
    for idx, (reference_benchmark, reference_label) in enumerate(zip(reference_benchmarks, reference_labels)):
        c = (0.3*(idx+1), 0.3*(idx+1), 0.3*(idx+1))
        ref_handles.append(ax.plot(np.log(reference_benchmark['n_neurons'].values),
                                   reference_benchmark['duration_run_rel']['amin'],
                                   '-o', label=reference_label, color=c,
                                   mec='white', ms=1)[0])
    ref_sizes = [reference_benchmark['n_neurons'].values[-1] for
                 reference_benchmark in reference_benchmarks]
    ref_sizes.extend([reference_benchmark['n_neurons'].values[-2] for reference_benchmark in reference_benchmarks])
    ref_sizes.extend([reference_benchmark['n_neurons'].values[-3] for reference_benchmark in reference_benchmarks])
    ref_sizes = sorted(ref_sizes)
    compare_sizes = []
    compare_times = []
    compare_labels = []
    for ref_size in ref_sizes:
        for reference_benchmark, reference_label in zip(reference_benchmarks, reference_labels):
            if ref_size in reference_benchmark['n_neurons'].values:
                ref_time = reference_benchmark.loc[reference_benchmark['n_neurons'] == ref_size]['duration_run_rel']['amin'].values
                assert len(ref_time) == 1
                compare_times.append(ref_time[0])
                compare_labels.append(reference_label + '({} neurons)'.format(ref_size))
                compare_sizes.append(ref_size)

    speedups = []
    algo_handles = []
    for idx, (benchmark, machine_name, GPU_name) in enumerate(zip(benchmarks,
                                                                  machine_names,
                                                                  GPU_names)):
        if idx not in select_benchmarks:
            continue
        gpu_results_pre = benchmark.loc[(benchmark['device'] == 'genn') &
                                        (benchmark['n_threads'] == 0) &
                                        (benchmark['algorithm'] == 'pre')]
        gpu_results_post = benchmark.loc[(benchmark['device'] == 'genn') &
                                         (benchmark['n_threads'] == 0) &
                                         (benchmark['algorithm'] == 'post')]
        gpu_results_pre = gpu_results_pre.sort_values(by='n_neurons')
        gpu_results_post = gpu_results_post.sort_values(by='n_neurons')
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
            style, label, mec, fc = '-', name, 'white', colors[idx]

        gpu_results = np.amin(np.vstack([gpu_results_pre['duration_run_rel']['amin'].values,
                                         gpu_results_post['duration_run_rel']['amin'].values]), axis=0)
        algorithm = np.argmin(np.vstack([gpu_results_pre['duration_run_rel']['amin'].values,
                                         gpu_results_post['duration_run_rel']['amin'].values]), axis=0)
        algo_handles.append(ax.plot(np.log(gpu_results_pre['n_neurons'].values),
                                    gpu_results,
                                    style, label=label, color=fc, mec=mec)[0])
        if not algorithm_details:
            ax.plot(np.log(gpu_results_pre['n_neurons'].values)[algorithm == 0],
                    gpu_results[algorithm == 0],
                    'o', label='_nolegend_', color=fc, mec=mec)
            ax.plot(np.log(gpu_results_pre['n_neurons'].values)[algorithm == 1],
                    gpu_results[algorithm == 1],
                    's', label='_nolegend_', color=fc, mec=mec, ms=5.5)
        this_speedup = []
        for compare_size, compare_time in zip(compare_sizes, compare_times):
            if compare_size in gpu_results_pre['n_neurons'].values:
                idx = np.nonzero(gpu_results_pre['n_neurons'] == compare_size)[0]
                this_speedup.append((compare_time/gpu_results[idx]).item())
            else:
                this_speedup.append(np.nan)
        speedups.append(this_speedup)

    if not algorithm_details:
        # Print out some values for the achieved speed up
        print('{}, speedups compared to: '.format(title))
        print(', '.join(compare_labels))
        for speedup, gpu in zip(speedups, gpu_names):
            print (gpu + ': ' + ', '.join('{:.1f}×'.format(s) for s in speedup))
        print()
    ax.grid(b=True, which='major', color='#e0e0e0', linestyle='-',
            linewidth=1.5)
    ax.grid(b=True, which='minor', color='#e0e0e0', linestyle='-',
            linewidth=0.5)
    ax.set_xticks(np.log(ticks))
    ax.set_xticklabels(ticks, rotation=45)
    ax.set(xlabel='Number of neurons',
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
                      ['“pre” strategy', '“post” strategy', 'best strategy'])
            ax.legend(algo_handles, labels, loc='upper left',
                      frameon=True, edgecolor='none', ncol=len(select_benchmarks),
                      columnspacing=0.)
    elif legend:
        ax.add_artist(plt.legend(loc='upper left', frameon=True, edgecolor='none'))
        from matplotlib.lines import Line2D
        from matplotlib.legend_handler import HandlerTuple
        second_legend = plt.legend([tuple(Line2D([], [], marker='o', color=c, ms=5.5, mec='none')
                                          for c in colors[:len(benchmarks)]),
                                    tuple(Line2D([], [], marker='s', color=c, mec='none')
                                          for c in colors[:len(benchmarks)])
                                    ], ['pre', 'post'],
                                   loc='lower right', frameon=True, edgecolor='none',
                                   numpoints=1,
                                   handler_map={tuple: HandlerTuple(ndivide=None)},
                                   title='best strategy:'
                                   )


def plot_necessary_runtime_across_gpus(benchmarks, reference_benchmark_cpu,
                                       reference_benchmark_gpu,
                                       labels, ax, title, ticks, legend=False,
                                       max_neurons=None, colors=None):
    if colors is None:
        colors = mpl.cm.tab10.colors
    used_neuron_values = set()
    # Make sure that the runtime was the same for all runs with the same
    # condition
    assert all(reference_benchmark_cpu['runtime']['std'] == 0)
    assert all(reference_benchmark_gpu['runtime']['std'] == 0)
    for idx, (benchmark, label) in enumerate(zip(benchmarks, labels)):
        benchmark = benchmark.loc[(benchmark['device'] == 'genn') &
                                  (benchmark['n_threads'] == 0)]
        # Make sure that the runtime was the same for all runs with the same
        # condition
        assert all(benchmark['runtime']['std'] == 0)
        # Merge the results from the two algorithms
        merged = pd.concat([benchmark['n_neurons'],
                            benchmark['total']['amin'],
                            benchmark['duration_run_rel']['amin'],
                            benchmark['duration_run']['amin']],
                           axis=1)
        merged.columns = ['n_neurons', 'total', 'duration_run_rel', 'duration_run']
        grouped = merged.groupby(['n_neurons'])
        benchmark = grouped.agg([np.min]).reset_index()
        benchmark = benchmark.sort_values(by='n_neurons')
        reference_benchmark_cpu = reference_benchmark_cpu.sort_values(by='n_neurons')
        # Only use those values where we have both kind of results
        available_sizes = set(benchmark['n_neurons'].unique()) & set(reference_benchmark_cpu['n_neurons'].unique())
        if len(set(benchmark['n_neurons'].unique()) - available_sizes):
            print('Benchmark {}/{} has no results for sizes {} on the CPU'.format(title, label, set(benchmark['n_neurons'].unique()) - available_sizes))
        if len(set(reference_benchmark_cpu['n_neurons'].unique()) - available_sizes):
            print('Benchmark {}/{} has no results for sizes {} on the GPU'.format(
                title, label, set(reference_benchmark_cpu['n_neurons'].unique()) - available_sizes))
        if max_neurons is not None:
            available_sizes = np.array(sorted(available_sizes))
            available_sizes = available_sizes[available_sizes <= max_neurons]
        benchmark = benchmark.loc[benchmark['n_neurons'].isin(available_sizes)]
        reference_benchmark_subset = reference_benchmark_cpu.loc[reference_benchmark_cpu['n_neurons'].isin(available_sizes)]
        reference_benchmark_subset_gpu = reference_benchmark_gpu.loc[reference_benchmark_gpu['n_neurons'].isin(available_sizes)]
        used_neuron_values |= set(benchmark['n_neurons'].values)
        variable_time_gpu = benchmark['duration_run_rel']['amin'].values
        fixed_time_gpu = reference_benchmark_subset_gpu['total']['amin'].values - reference_benchmark_subset_gpu['duration_run']['amin'].values
        variable_time_cpu = reference_benchmark_subset['duration_run_rel']['amin'].values
        fixed_time_cpu = reference_benchmark_subset['total']['amin'].values - reference_benchmark_subset['duration_run']['amin'].values
        # Check assumptions
        necessary = (fixed_time_cpu - fixed_time_gpu)/(variable_time_gpu - variable_time_cpu)
        # If GPU takes longer per simulated second, no way to get a faster sim
        necessary[variable_time_gpu > variable_time_cpu] = np.NaN
        # Fixed time is already lower for GPU
        necessary[fixed_time_gpu < fixed_time_cpu] = 0
        if any(fixed_time_gpu < fixed_time_cpu):
            print('Fixed time on GPU is lower for', label)
        ax.plot(np.log(benchmark['n_neurons']).unique(), necessary, 'o-',
                mec='white', label=label, color=colors[idx])

    ax.set_xticklabels(ticks, rotation=45)
    ax.grid(b=True, which='major', color='#e0e0e0', linestyle='-',
            linewidth=1.5)
    ax.grid(b=True, which='minor', color='#e0e0e0', linestyle='-',
            linewidth=0.5)
    ax.set(xticks=np.log(ticks),
           xlabel='Number of neurons',
           ylabel='necessary biological runtime (s)',
           yscale='log', title=title)
    if legend:
        ax.legend(loc='lower left', frameon=True, edgecolor='none')


if __name__ == '__main__':
    plt.style.use('figures.conf')
    target_dir = 'benchmark_results/comparisons'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    benchmark_dirs = ['benchmark_results/2018-10-05_vuvuzela',
                      'benchmark_results/2018-10-02_inf900777',
                      'benchmark_results/2018-10-02_f152b85d2726',
                      'benchmark_results/2018-10-04_jwc09n012']

    float_dtypes_per_benchmark = [('float32', ),
                                  ('float64', ),
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
                             figsize=(6.33, 6.33*1))
    fig_gpu, axes_gpu = plt.subplots(2, 2, sharey='row', sharex='row',
                                     figsize=(6.33, 6.33*1))

    for col, float_dtype in enumerate(['float64', 'float32']):
        precision = 'single precision' if float_dtype == 'float32' else 'double precision'
        precision_short = 'single' if float_dtype == 'float32' else 'double'
        for ax, title, fname, ticks in [(axes[1, col], 'COBAHH', 'benchmarks_COBAHH.txt', COBAHH_xticks),
                                        (axes[0, col], 'Mbody', 'benchmarks_Mbody_example.txt', MBody_xticks)]:
            benchmarks = [mean_and_std_fixed_time(load_benchmark(dirname, fname),
                                                  monitor=False, float_dtype=float_dtype)
                          for dirname in benchmark_dirs]

            plot_total_comparisons(benchmarks, machine_names, gpu_names,
                                   ax, title + ' – ' + precision,
                                   ticks, legend=(ax == axes[0, 1]))

        for ax, title, fname, ticks in [(axes_gpu[1, col], 'COBAHH', 'benchmarks_COBAHH.txt', COBAHH_xticks),
                                 (axes_gpu[0, col], 'Mbody', 'benchmarks_Mbody_example.txt', MBody_xticks)]:
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
                                            ticks=ticks,
                                            legend=(ax == axes_gpu[1, 1]),
                                            colors=mpl.cm.tab10.colors[:3] + mpl.cm.tab10.colors[4:])  # avoid red-green
    for ax in [axes_gpu[0, 1], axes_gpu[1, 1], axes[0, 1], axes[1, 1]]:
        ax.set_ylabel(None)
    fig.tight_layout()
    fig.savefig(os.path.join(target_dir,
                             'runtime_comparison_{}'.format(FIGURE_EXTENSION)))
    plt.close(fig)

    fig_gpu.tight_layout()
    fig_gpu.savefig(os.path.join(target_dir,
                             'gpu_runtime_comparison{}'.format(FIGURE_EXTENSION)))
    plt.close(fig_gpu)

    fig_gpu_algos, axes_gpu_algos = plt.subplots(1, 2, sharey='row',
                                                 figsize=(6.33, 6.33*0.5))
    float_dtype = 'float32'
    for ax_detail, title, fname, ticks in [(axes_gpu_algos[1],
                                            'COBAHH', 'benchmarks_COBAHH.txt', COBAHH_xticks),
                                    (axes_gpu_algos[0],
                                     'Mbody', 'benchmarks_Mbody_example.txt', MBody_xticks)]:
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
                                        ticks=ticks,
                                        legend=(ax_detail == axes_gpu_algos[0]),
                                        algorithm_details=True,
                                        select_benchmarks=[2, 3],
                                        colors=mpl.cm.tab10.colors[:3] + mpl.cm.tab10.colors[4:])  # avoid red-green
    axes_gpu_algos[1].set_ylabel(None)
    fig_gpu_algos.tight_layout()
    fig_gpu_algos.savefig(os.path.join(target_dir,
                                       'gpu_runtime_comparison_algos{}'.format(FIGURE_EXTENSION)))
    plt.close(fig_gpu_algos)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey='row',
                                            figsize=(6.33, 6.33*0.5))
    for ax, title, fname, max_neurons, ticks in [(ax_right, 'COBAHH', 'benchmarks_COBAHH.txt', None, COBAHH_xticks),
                                          (ax_left, 'Mbody', 'benchmarks_Mbody_example.txt', None, MBody_xticks)]:
        benchmarks = [mean_and_std_fixed_time(load_benchmark(dirname, fname),
                                              monitor=False, float_dtype=float_dtype)
                      for dirname, float_dtypes in zip(benchmark_dirs, float_dtypes_per_benchmark)
                      for float_dtype in float_dtypes]

        reference = mean_and_std_fixed_time(load_benchmark(reference_dir, fname),
                                            monitor=False,
                                            float_dtype='float64')
        reference_cpu = reference.loc[(reference['device'] == 'cpp_standalone') &
                                  (reference['n_threads'] == 24)]
        reference_gpu = reference.loc[(reference['device'] == 'genn') &
                                      (reference['n_threads'] == 0) &
                                      (reference['algorithm'] == 'pre')]  # arbitrary
        labels = ['%s (%s)' % (gpu_name, 'single' if float_dtype == 'float32' else 'double')
                  for gpu_name, float_dtypes in zip(gpu_names, float_dtypes_per_benchmark)
                  for float_dtype in float_dtypes]
        plot_necessary_runtime_across_gpus(benchmarks, reference_cpu, reference_gpu,
                                           labels,
                                           ax, ticks=ticks,
                                           legend=(ax == ax_right),
                                           title=title, max_neurons=max_neurons,
                                           colors=mpl.cm.tab10.colors[:3] + mpl.cm.tab10.colors[4:])  # avoid red-green)
    ax_right.set_ylabel(None)
    fig.tight_layout()
    fname = os.path.join(target_dir,
                         'necessary_biological_runtime_across_GPUs{}{}'.format(monitor_str,
                                                                               FIGURE_EXTENSION))
    fig.savefig(fname)
    plt.close(fig)
