# coding=utf-8
from __future__ import unicode_literals

import os
import sys
import itertools

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

FIGURE_EXTENSION = '.png'

plt.style.use('figures.conf')

directory = None  # replace this or pass the directory as a command line arg


def load_benchmark(directory, fname):
    full_fname = os.path.join(directory, fname)
    benchmarks = pd.read_csv(full_fname, sep=r'\s+', header=None, index_col=None,
                             names=['device', 'n_threads', 'n_neurons', 'n_synapses',
                                    'runtime', 'with_monitor', 'float_dtype',
                                    'total',
                                    't_after_load',
                                    't_before_synapses', 't_after_synapses',
                                    't_after_init', 't_before_run',
                                    't_after_run', 't_before_write', 't_after_write'],
                             dtype={'device': 'category',
                                    'with_monitor': 'bool',
                                    'float_dtype': 'category',
                                    'runtime': 'float64',
                                    'total': 'float64'})
    # The times in the benchmark file are the full times (in milliseconds) that
    # have elapsed since the start of the simulation.
    # codegen & build time is the time in the total time that was not measured by GeNN
    benchmarks['duration_compile'] = benchmarks['total'] - benchmarks['t_after_write'] / 1000.  # t_after_write is last measured time point

    # Prepare time includes allocating memory and loading static arrays from disk
    # In GeNN, this also includes things like converting arrays to GeNN's format
    benchmarks['duration_before'] = (benchmarks['t_after_load'] +
                                     (benchmarks['t_before_run'] -
                                      benchmarks['t_after_init'])) / 1000.

    # Synapse creation
    benchmarks['duration_synapses'] = (benchmarks['t_after_synapses'] -
                                       benchmarks['t_before_synapses']) / 1000.

    # Neuronal + synaptic variable initialization
    benchmarks['duration_init'] = (benchmarks['t_after_init'] -
                                   benchmarks['t_after_synapses']) / 1000.

    # The actual simulation time
    benchmarks['duration_run'] = (benchmarks['t_after_run'] -
                                  benchmarks['t_before_run']) / 1000.
    # Simulation time relative to realtime
    benchmarks['duration_run_rel'] = benchmarks['duration_run']/benchmarks['runtime']

    # The time after the simulation, most importantly to write values to disk
    # but for Brian2GeNN also conversion from GeNN data structures to Brian format
    benchmarks['duration_after'] = (benchmarks['t_after_write'] -
                                    benchmarks['t_after_run']) / 1000.

    return benchmarks


def mean_and_std_fixed_time(benchmarks, monitor=True, float_dtype='float64',
                            runtime=1):
    subset = benchmarks.loc[(benchmarks['with_monitor'] == monitor) &
                            (benchmarks['runtime'] == runtime) &
                            (benchmarks['float_dtype'] == float_dtype)]

    # Average over trials
    grouped = subset.groupby(['device', 'n_neurons', 'n_threads'])
    aggregated = grouped.agg([np.min, np.mean, np.std]).reset_index()

    return aggregated


def label_and_color(device, n_threads, all_threads):
    if device == 'genn':
        if n_threads == -1:
            return 'Brian2GeNN CPU', 'lightblue'
        else:
            return 'Brian2GeNN GPU', 'darkblue'
    else:
        label = 'C++ %2d threads' % n_threads
        colors = ['gold', 'darkorange', 'tomato', 'darkred',
                  'darkviolet', 'indigo']
        color = colors[np.nonzero(sorted(all_threads) == n_threads)[0].item()]
        return label, color


def inside_title(ax, text, x=0.98, y=.04, horizontalalignment='right'):
    t = ax.text(x, y, text, weight='bold', horizontalalignment=horizontalalignment,
                     transform=ax.transAxes)
    t.set_bbox({'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8})


def plot_total(benchmarks, ax, legend=False, skip=('Brian2GeNN CPU',),
               plot_what='total',
               axis_label='Total wall clock time (including compilation)'):
    ax.set_yscale('log')
    # We do the log scale for the x axis manually -- easier to get the ticks/labels right
    conditions = benchmarks.groupby(['device', 'n_threads'])
    all_threads = benchmarks.loc[benchmarks['device'] == 'cpp_standalone']['n_threads'].unique()
    for condition in conditions:
        (device, threads), results = condition
        label, color = label_and_color(device, threads, all_threads)
        if label in skip:
            continue
        ax.plot(np.log(results['n_neurons'].values),
                results[plot_what]['amin'],
                'o-', label=label, color=color, mec='none')
    if legend:
        ax.legend(loc='upper left', frameon=True, edgecolor='none')
    used_n_neuron_values = benchmarks['n_neurons'].unique()
    # Make sure we show the xtick label for the highest value
    if len(used_n_neuron_values) % 2 == 0:
        start = 1
    else:
        start = 0
    ax.set_xticks(np.log(sorted(used_n_neuron_values))[start::2])
    ax.set_xticklabels(sorted(used_n_neuron_values)[start::2], rotation=45)
    ax.set(xlabel='Model size (# neurons)',
           ylabel=axis_label)


def plot_total_comparisons(benchmarks, machine_names, GPU_names, ax, legend=False):
    colors = mpl.cm.tab10.colors
    for idx, (benchmark, machine_name, GPU_name) in enumerate(zip(benchmarks,
                                                                  machine_names,
                                                                  GPU_names)):
        # We do the log scale for the x axis manually -- easier to get the ticks/labels right
        # Only use Brian2GeNN GPU and maximum number of threads
        max_threads = benchmark.loc[benchmark['device'] == 'cpp_standalone']['n_threads'].max()
        gpu_results = benchmark.loc[(benchmark['device'] == 'genn') & (benchmark['n_threads'] == 0)]
        cpu_results = benchmark.loc[(benchmark['device'] == 'cpp_standalone') & (benchmark['n_threads'] == max_threads)]

        for subset, name, ls in [(gpu_results, '{} – {}'.format(machine_name, GPU_name), ':'),
                                 (cpu_results, '{} – {} CPU cores'.format(machine_name, max_threads), ':')]:
            if len(subset) == 0:
                continue
            ax.plot(np.log(subset['n_neurons'].values),
                    subset['duration_run']['amin'],
                    'o-', label=name, color=colors[idx], linestyle=ls, mec='none')
        used_n_neuron_values = benchmark['n_neurons'].unique()
        # Make sure we show the xtick label for the highest value
    if len(used_n_neuron_values) % 2 == 0:
        start = 1
    else:
        start = 0
    ax.set_xticks(np.log(sorted(used_n_neuron_values))[start::2])
    ax.set_xticklabels(sorted(used_n_neuron_values)[start::2], rotation=45)
    ax.set(xlabel='Model size (# neurons)',
           ylabel='Simulation time (s)',
           yscale='log')
    if legend:
        ax.legend(loc='upper left', frameon=True, edgecolor='none')


def plot_detailed_times(benchmark, ax_detail, ax_sim, title=None, legend=False):
    # Prepare data for stacked plot
    x = np.array(benchmark['n_neurons'])
    # code generation and build
    build = benchmark['duration_compile']['amin']
    # Reading/writing arrays from/to disk / conversion between Brian2 and GeNN format
    read_write_convert = (benchmark['duration_before']['amin'] +
                          benchmark['duration_after']['amin'])
    # Creating synapses and initialization
    create_initialize = (benchmark['duration_synapses']['amin'] +
                         benchmark['duration_init']['amin'])
    # Simulation time
    simulate = benchmark['duration_run']['amin']
    # TODO: Check that the total matches
    total = benchmark['total']['amin']
    handles = []
    labels = []
    for data, label in [(build, 'code gen & compile'),
                        (read_write_convert, 'overhead'),
                        (create_initialize, 'synapse creation & initialization')]:
        h = ax_detail.plot(np.log(x), data, 'o-', mec='white')[0]
        if legend:
            xl, yl = legend[label]
            ax_detail.text(np.log(xl), yl, label,
                           fontsize=8, fontweight='bold', color=h.get_color())
    ax_detail.grid(b=True, which='major', color='#b0b0b0', linestyle='-',
                   linewidth=0.5)
    # We plot simulation/total times for 1s, 10s, and 100s
    for idx, (biological_time, align) in enumerate([(1, 'top'),
                                                    (10, 'center'),
                                                    (100, 'bottom')]):
        import colorsys
        brighten_by = idx * 0.2
        basecolor_sim = colorsys.rgb_to_hls(*mpl.cm.tab10.colors[3])
        color_sim = colorsys.hls_to_rgb(basecolor_sim[0],
                                        basecolor_sim[1] + brighten_by*(1 - basecolor_sim[1]),
                                        basecolor_sim[2])
        ax_sim.plot(np.log(x), simulate*biological_time,
                    'o-', mec='white', color=color_sim)
        if idx == 0:
            if legend:
                xl, yl = legend['simulation']
                ax_sim.text(np.log(xl), yl, 'simulation',
                            fontsize=8, color=color_sim,
                            fontweight='bold')

        ax_sim.text(np.log(x[-1]), simulate.values[-1]*biological_time,
                    ' {:3d}s'.format(biological_time),
                    verticalalignment=align, fontsize=8, color=color_sim,
                    fontweight='bold')
    ax_sim.grid(b=True, which='major', color='#b0b0b0', linestyle='-',
                   linewidth=0.5)
    # Make sure we show the xtick label for the highest value
    if len(x) % 2 == 0:
        start = 1
    else:
        start = 0
    ax_detail.set(xticks=np.log(x)[start::2],
                  xlabel='Model size (# neurons)',
                  ylabel='Wall clock time (s)')
    ax_sim.set(xlabel='', ylabel='', yscale='log')
    ax_sim.set_yticks([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    ax_sim.spines['bottom'].set_visible(False)
    ax_detail.set_yscale('symlog', linthreshy=0.01, linscaley=np.log10(np.e))
    ax_detail.set_xticklabels(sorted(x)[start::2], rotation=45)
    plt.setp(ax_sim.xaxis.get_ticklines(), visible=False)
    if title is not None:
        ax_sim.set_title(title)


def plot_detailed_times_stacked(benchmark, ax, legend=False):
    # Prepare data for stacked plot
    x = np.array(benchmark['n_neurons'])
    # code generation and build
    build = benchmark['duration_compile']['amin']
    # Reading/writing arrays from/to disk / conversion between Brian2 and GeNN format
    read_write_convert = (benchmark['duration_before']['amin'] +
                          benchmark['duration_after']['amin'])
    # Creating synapses and initialization
    create_initialize = (benchmark['duration_synapses']['amin'] +
                         benchmark['duration_init']['amin'])
    # Simulation time
    simulate = benchmark['duration_run']['amin']
    # TODO: Check that the total matches
    total = benchmark['total']['amin']

    # Normalize everything to total time and stack up
    Y = np.row_stack([build/total,
                      read_write_convert/total,
                      create_initialize/total,
                      simulate/total])

    labels = ['code gen & compile',
              'overhead',
              'synapse creation & initialization',
              'simulation']
    handles = ax.stackplot(np.log(x), *(Y*100),
                           labels=labels)

    ax.grid(b=True, which='minor', color='#b0b0b0', linestyle='-',
            linewidth=0.33)
    if legend:
        # Show legend in reverse order to avoid confusion
        ax.legend(reversed(handles), reversed(labels),
                  bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                  mode="expand", borderaxespad=0)
    # Make sure we show the xtick label for the highest value
    if len(x) % 2 == 0:
        start = 1
    else:
        start = 0
    ax.set(xticks=np.log(x)[start::2],
           xlabel='Model size (# neurons)',
           ylabel='% of total time',
           ylim=(0, 100))
    ax.set_xticklabels(sorted(x)[start::2], rotation=45)


def plot_detail_comparison(benchmarks):
    fig, axes = plt.subplots(2, 2, sharex='all', figsize=(6.3, 6.3))
    for float_dtype, with_monitor in itertools.product(['float32', 'float64'],
                                                       [False, True]):
        setup_type = '{} precision ({} monitor)'.format('single' if float_dtype == 'float32' else 'double',
                                                        'with' if with_monitor else 'no')
        benchmark = benchmarks.loc[(benchmarks['float_dtype'] == float_dtype) &
                                   (benchmarks['with_monitor'] == with_monitor)]
        if not len(benchmark):
            continue
        # Prepare data for stacked plot
        x = np.array(benchmark['n_neurons'])
        # code generation and build
        build = benchmark['duration_compile']['amin']
        # Reading/writing arrays from/to disk / conversion between Brian2 and GeNN format
        read_write_convert = (benchmark['duration_before']['amin'] +
                              benchmark['duration_after']['amin'])
        # Creating synapses and initialization
        create_initialize = (benchmark['duration_synapses']['amin'] +
                             benchmark['duration_init']['amin'])
        # Simulation time
        simulate = benchmark['duration_run']['amin']
        # TODO: Check that the total matches
        total = benchmark['total']['amin']

        labels = ['code gen & compile',
                  'overhead',
                  'synapse creation & initialization',
                  'simulation']

        for ax, label, data in zip(axes.flat, labels,
                                   [build, read_write_convert,
                                    create_initialize, simulate]):
            ax.plot(np.log(x), data, 'o-', label=setup_type, mec='white')
            ax.grid(b=True, which='minor', color='#b0b0b0', linestyle='-',
                    linewidth=0.33)
            # Make sure we show the xtick label for the highest value
            if len(x) % 2 == 0:
                start = 1
            else:
                start = 0
            ax.set(xticks=np.log(x)[start::2],
                   xlabel='Model size (# neurons)',
                   ylabel='%s time (s)' % label,
                   yscale='symlog')
            ax.set_xticklabels(sorted(x)[start::2], rotation=45)
    for ax in axes.flat:
        ax.set_ylim(ymin=0)
    axes[0, 0].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                      mode="expand", borderaxespad=0)
    plt.tight_layout()
    return fig


def plot_necessary_runtime(benchmarks, reference_benchmarks, labels, ax, legend=False):
    for benchmark, reference_benchmark, label in zip(benchmarks, reference_benchmarks, labels):
        benchmark.sort_values(by='n_neurons')
        reference_benchmark.sort_values(by='n_neurons')
        variable_time_gpu = benchmark['duration_run']['amin'].values
        fixed_time_gpu = benchmark['total']['amin'].values - variable_time_gpu
        variable_time_cpu = reference_benchmark['duration_run']['amin'].values
        fixed_time_cpu = reference_benchmark['total']['amin'].values - variable_time_cpu
        # Check assumptions
        assert all(fixed_time_gpu > fixed_time_cpu)
        necessary = (fixed_time_cpu - fixed_time_gpu)/(variable_time_gpu - variable_time_cpu)
        # If GPU takes longer per simulated second, no way to get a faster sim
        necessary[variable_time_gpu > variable_time_cpu] = np.NaN

        ax.plot(np.log(benchmark['n_neurons']).values, necessary, 'o-',
                mec='white', label=label)

    # Make sure we show the xtick label for the highest value
    x = benchmarks[0]['n_neurons']
    if len(x) % 2 == 0:
        start = 1
    else:
        start = 0

    ax.set_xticklabels(sorted(x)[start::2], rotation=45)
    ax.grid(b=True, which='minor', color='#b0b0b0', linestyle='-',
            linewidth=0.33)
    ax.set(xticks=np.log(x)[start::2],
           xlabel='Model size (# neurons)',
           ylabel='necessary biological runtime (s)',
           yscale='log')
    if legend:
        ax.legend(loc='upper right', frameon=True, edgecolor='none')


if __name__ == '__main__':
    if directory is None:
        if len(sys.argv) == 2:
            directory = sys.argv[1]
        else:
            raise ValueError('Need the directory name as an argument')

    COBA_full = load_benchmark(directory, 'benchmarks_COBAHH.txt')
    Mbody_full = load_benchmark(directory, 'benchmarks_Mbody_example.txt')

    # Summary plot, showing the biological runtime that is necessary to get a
    # faster simulation with the GPU
    COBA32 = mean_and_std_fixed_time(COBA_full, monitor=False, float_dtype='float32')
    COBA64 = mean_and_std_fixed_time(COBA_full, monitor=False, float_dtype='float64')
    Mbody32 = mean_and_std_fixed_time(Mbody_full, monitor=False, float_dtype='float32')
    Mbody64 = mean_and_std_fixed_time(Mbody_full, monitor=False, float_dtype='float64')
    benchmark_gpus = [COBA32.loc[(COBA32['device'] == 'genn') &
                                 (COBA32['n_threads'] == 0)],
                      COBA64.loc[(COBA64['device'] == 'genn') &
                                 (COBA64['n_threads'] == 0)],
                      Mbody32.loc[(Mbody32['device'] == 'genn') &
                                  (Mbody32['n_threads'] == 0)],
                      Mbody64.loc[(Mbody64['device'] == 'genn') &
                                  (Mbody64['n_threads'] == 0)]
                      ]
    max_threads = COBA32.loc[COBA32['device'] == 'cpp_standalone']['n_threads'].max()
    benchmark_cpus = [COBA32.loc[(COBA32['device'] == 'cpp_standalone') &
                                 (COBA32['n_threads'] == max_threads)],
                      COBA64.loc[(COBA64['device'] == 'cpp_standalone') &
                                 (COBA64['n_threads'] == max_threads)],
                      Mbody32.loc[(Mbody32['device'] == 'cpp_standalone') &
                                  (Mbody32['n_threads'] == max_threads)],
                      Mbody64.loc[(Mbody64['device'] == 'cpp_standalone') &
                                  (Mbody64['n_threads'] == max_threads)]
                      ]
    labels = ['COBA – single precision', 'COBA – double precision',
              'Mbody – single precision', 'Mbody – double precision']

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6.3, 6.3*.6666),
                                            sharey='row')
    plot_necessary_runtime(benchmark_gpus[:2], benchmark_cpus[:2], labels[:2], ax_right, legend=True)
    plot_necessary_runtime(benchmark_gpus[2:], benchmark_cpus[2:], labels[2:], ax_left, legend=True)

    fig.tight_layout()
    fig.savefig(os.path.join(directory,
                             'necessary_biological_runtime' + FIGURE_EXTENSION))
    plt.close(fig)

    for with_monitor in [True, False]:
        monitor_suffix = '' if with_monitor else '_no_monitor'
        # COBA with linear scaling
        COBA = mean_and_std_fixed_time(COBA_full, monitor=with_monitor)
        COBA32 = mean_and_std_fixed_time(COBA_full, monitor=with_monitor,
                                         float_dtype='float32')
        # Mushroom body
        Mbody = mean_and_std_fixed_time(Mbody_full, monitor=with_monitor)
        Mbody32 = mean_and_std_fixed_time(Mbody_full, monitor=with_monitor,
                                          float_dtype='float32')

        for COBA_benchmark, Mbody_benchmark, suffix in [(COBA, Mbody, ''),
                                                        (COBA32, Mbody32, ' single precision')]:
            # Total time (including compilation)
            fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey='row',
                                                    figsize=(6.3, 6.3 * .666))
            plot_total(COBA_benchmark, ax_left, legend=True)
            ax_left.set_title('COBAHH%s' % suffix)
            plot_total(Mbody_benchmark, ax_right, legend=False)
            ax_right.set_title('Mushroom body%s' % suffix)
            plt.tight_layout()
            fig.savefig(os.path.join(directory, 'total_runtime%s%s%s' % (monitor_suffix,
                                                                         suffix,
                                                                         FIGURE_EXTENSION)))

            # Runtime relative to realtime
            fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey='row',
                                                    figsize=(6.3, 6.3 * .666))
            plot_total(COBA_benchmark, ax_left, legend=True, plot_what='duration_run_rel',
                       axis_label='Simulation time (relative to real-time)')
            ax_left.set_title('COBAHH')
            plot_total(Mbody_benchmark, ax_right, legend=False, plot_what='duration_run_rel',
                       axis_label='Simulation time (relative to real-time)')
            ax_right.set_title('Mushroom body')
            plt.tight_layout()
            fig.savefig(os.path.join(directory, 'simulation_time_only%s%s%s' % (monitor_suffix,
                                                                                suffix,
                                                                                FIGURE_EXTENSION)))

    # TODO: This is a bit redundant...
    max_threads = COBA_full.loc[COBA_full['device'] == 'cpp_standalone']['n_threads'].max()
    for name, dev, threads in [('cpp', 'cpp_standalone', max_threads),
                               ('GeNN GPU', 'genn', 0 )]:
        subset = COBA_full.loc[(COBA_full['device'] == dev) &
                               (COBA_full['n_threads'] == threads)]
        grouped = subset.groupby(['n_neurons', 'float_dtype', 'with_monitor'])
        COBA_all = grouped.agg([np.min, np.mean, np.std]).reset_index()

        subset = Mbody_full.loc[(Mbody_full['device'] == dev) &
                                (Mbody_full['n_threads'] == threads)]
        grouped = subset.groupby(['n_neurons', 'float_dtype', 'with_monitor'])
        Mbody_all = grouped.agg([np.min, np.mean, np.std]).reset_index()

        fig = plot_detail_comparison(COBA_all)
        fig.savefig(os.path.join(directory, 'COBA_detail_comparison%s' % FIGURE_EXTENSION))

        fig = plot_detail_comparison(Mbody_all)
        fig.savefig(
            os.path.join(directory, 'Mbody_detail_comparison%s' % FIGURE_EXTENSION))

        for benchmarks, name in [(COBA, 'COBAHH'),
                                 (Mbody, 'Mbody'),
                                 (COBA32, 'COBAHH single precision'),
                                 (Mbody32, 'Mbody single precision')]:
            # We use the maximum number of threads for which we have data
            max_threads = benchmarks.loc[benchmarks['device'] == 'cpp_standalone']['n_threads'].max()
            fig, axes = plt.subplots(2, 2, sharey='row', sharex='col',
                                     figsize=(6.3, 6.3 * .666),
                                     gridspec_kw={'height_ratios': [1, 4]})
            GeNN_GPU = benchmarks.loc[(benchmarks['device'] == 'genn') &
                                      (benchmarks['n_threads'] == 0)]
            if len(GeNN_GPU):
                plot_detailed_times(GeNN_GPU, axes[1, 0], axes[0, 0],
                                    title='%s (Brian2GeNN GPU)' % name,
                                    legend={'simulation': (300, 1e3),
                                            'code gen & compile': (300, 20),
                                            'overhead': (300, 0.9),
                                            'synapse creation & initialization': (2700, 1.1e-2)})
            else:
                ax_left.axis('off')
            cpp_threads = benchmarks.loc[(benchmarks['device'] == 'cpp_standalone') &
                                         (benchmarks['n_threads'] == max_threads)]
            if len(cpp_threads):
                plot_detailed_times(cpp_threads, axes[1, 1], axes[0, 1],
                                    title='%s (C++ %d threads)' % (name, max_threads))
            else:
                ax_right.axis('off')
            plt.tight_layout()
            if len(GeNN_GPU) or len(cpp_threads):
                fig.savefig(os.path.join(directory,
                                         'detailed_runtime_%s%s%s' % (name,
                                                                      monitor_suffix,
                                                                      FIGURE_EXTENSION)))

            fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey='row',
                                                    figsize=(6.3, 6.3 * .666))
            if len(GeNN_GPU):
                plot_detailed_times_stacked(GeNN_GPU, ax_left, legend=True)
                inside_title(ax_left, '%s (Brian2GeNN GPU)' % name)
            else:
                ax_left.axis('off')
            if len(cpp_threads):
                plot_detailed_times_stacked(cpp_threads, ax_right)
                inside_title(ax_right, '%s (C++ %d threads)' % (name, max_threads))
            else:
                ax_right.axis('off')
            plt.tight_layout()
            if len(GeNN_GPU) or len(cpp_threads):
                fig.savefig(os.path.join(directory,
                                         'detailed_runtime_relative_%s%s%s' % (name,
                                                                               monitor_suffix,
                                                                               FIGURE_EXTENSION)))

        # Plot details for GPU-only and a single benchmark
        for benchmark_double, benchmark_float, name in [(COBA, COBA32, 'COBAHH'),
                                                        (Mbody, Mbody32, 'Mbody')]:
            gpu = benchmark_double.loc[(benchmark_double['device'] == 'genn') &
                                       (benchmark_double['n_threads'] == 0)]
            gpu32 = benchmark_float.loc[(benchmark_float['device'] == 'genn') &
                                        (benchmark_float['n_threads'] == 0)]

            fig, axes = plt.subplots(2, 2, sharey='row', sharex='col',
                                     figsize=(6.3, 6.3 * .666),
                                     gridspec_kw={'height_ratios': [1, 2]})
            plot_detailed_times(gpu, axes[1, 0], axes[0, 0],
                                legend={'simulation': (300, 2e2),
                                        'code gen & compile': (300, 20),
                                        'overhead': (300, 0.9),
                                        'synapse creation & initialization': (2700, 1.1e-2)},
                                title=name + ' – double precision')
            plot_detailed_times(gpu32, axes[1, 1], axes[0, 1],
                                title=name + ' – single precision')
            fig.tight_layout()
            fig.savefig(os.path.join(directory,
                                     'gpu_detailed_runtime_%s%s' % (name, FIGURE_EXTENSION)))
            plt.close(fig)

