# coding=utf-8
from __future__ import unicode_literals

import os
import sys
import itertools

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

FIGURE_EXTENSION = '.pdf'

plt.style.use('figures.conf')

directory = 'benchmark_results/2018-10-02_f152b85d2726'  # replace this or pass the directory as a command line arg

# Hardcoded to easily get consistent xticks over all plots
MBody_xticks = [325, 825, 5200, 40200, 320200, 2560200, 20480200]
COBAHH_xticks = [400, 2000, 8000, 32000, 128000, 512000, 2048000]

def load_benchmark(directory, fname):
    full_fname = os.path.join(directory, fname)
    benchmarks = pd.read_csv(full_fname, sep=r'\s+', header=None, index_col=None,
                             names=['device', 'algorithm', 'n_threads', 'n_neurons', 'n_synapses',
                                    'runtime', 'with_monitor', 'float_dtype',
                                    'total',
                                    't_after_load',
                                    't_before_synapses', 't_after_synapses',
                                    't_after_init', 't_before_run',
                                    't_after_run', 't_before_write', 't_after_write'],
                             dtype={'device': 'category',
                                    'algorithm': 'category',
                                    'with_monitor': 'bool',
                                    't_after_load': 'int64',
                                    't_before_synapses': 'int64',
                                    't_after_synapses': 'int64',
                                    't_after_init': 'int64',
                                    't_before_run': 'int64',
                                    't_after_run': 'int64',
                                    't_before_write': 'int64',
                                    't_after_write': 'int64',
                                    'float_dtype': 'category',
                                    'runtime': 'float64',
                                    'total': 'float64'})
    benchmarks.replace(np.nan, 'N/A', inplace=True)
    # The times in the benchmark file are the full times (in microseconds) that
    # have elapsed since the start of the simulation.
    # codegen & build time is the time in the total time that was not measured by GeNN
    benchmarks['duration_compile'] = benchmarks['total'] - benchmarks['t_after_write'] / 1e6  # t_after_write is last measured time point

    # Prepare time includes allocating memory and loading static arrays from disk
    # In GeNN, this also includes things like converting arrays to GeNN's format
    benchmarks['duration_before'] = (benchmarks['t_after_load'] +
                                     (benchmarks['t_before_run'] -
                                      benchmarks['t_after_init'])) / 1e6

    # Synapse creation
    benchmarks['duration_synapses'] = (benchmarks['t_after_synapses'] -
                                       benchmarks['t_before_synapses']) / 1e6

    # Neuronal + synaptic variable initialization
    benchmarks['duration_init'] = (benchmarks['t_after_init'] -
                                   benchmarks['t_after_synapses']) / 1e6

    # The actual simulation time
    benchmarks['duration_run'] = (benchmarks['t_after_run'] -
                                  benchmarks['t_before_run']) / 1e6
    # Simulation time relative to realtime
    benchmarks['duration_run_rel'] = benchmarks['duration_run']/benchmarks['runtime']

    # The time after the simulation, most importantly to write values to disk
    # but for Brian2GeNN also conversion from GeNN data structures to Brian format
    benchmarks['duration_after'] = (benchmarks['t_after_write'] -
                                    benchmarks['t_after_run']) / 1e6

    return benchmarks


def mean_and_std_fixed_time(benchmarks, monitor=True, float_dtype='float64'):
    subset = benchmarks.loc[(benchmarks['with_monitor'] == monitor) &
                            (benchmarks['float_dtype'] == float_dtype)]

    # Average over trials
    grouped = subset.groupby(['device', 'algorithm', 'n_neurons', 'n_threads', 'runtime'])
    aggregated = grouped.agg([np.min, np.mean, np.std]).reset_index()

    return aggregated


def label_and_color(device, algorithm, n_threads, all_threads):
    if device == 'genn':
        if n_threads == -1:
            label, color, linestyle = 'Brian2GeNN CPU', 'lightblue', 'o-'
        else:
            label, color = 'Brian2GeNN GPU', 'darkblue'
            if algorithm == 'pre':
                label += ' (pre)'
                linestyle = ':'
            else:
                label += ' (post)'
                linestyle = '--'
    else:
        label = 'C++ %2d threads' % n_threads
        colors = ['gold', 'darkorange', 'tomato', 'darkred',
                  'darkviolet', 'indigo']
        color = colors[np.nonzero(sorted(all_threads) == n_threads)[0].item()]
        linestyle = 'o-'

    return label, color, linestyle


def inside_title(ax, text, x=0.98, y=.04, horizontalalignment='right'):
    t = ax.text(x, y, text, weight='bold', horizontalalignment=horizontalalignment,
                     transform=ax.transAxes)
    t.set_bbox({'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8})


def plot_total(benchmarks, ax, legend=False, skip=('Brian2GeNN CPU',),
               plot_what='total',
               axis_label='Total wall clock time (including compilation)'):
    ax.set_yscale('log')
    # We do the log scale for the x axis manually -- easier to get the ticks/labels right
    conditions = benchmarks.groupby(['device', 'algorithm', 'n_threads'])
    all_threads = benchmarks.loc[benchmarks['device'] == 'cpp_standalone']['n_threads'].unique()
    for condition in conditions:
        (device, algorithm, threads), results = condition
        label, color, linestyle = label_and_color(device, algorithm, threads, all_threads)
        if label in skip:
            continue
        ax.plot(np.log(results['n_neurons'].values),
                results[plot_what]['amin'],
                linestyle, label=label, color=color, mec='none')
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


def plot_detailed_times(benchmark, ax_detail, ax_sim, ticks, title=None, legend=False):
    if len(benchmark['algorithm'].unique()) > 1:
        benchmark = pd.concat([benchmark['n_neurons'],
                               benchmark['runtime'],
                               benchmark['duration_compile']['amin'],
                               benchmark['duration_before']['amin'],
                               benchmark['duration_after']['amin'],
                               benchmark['duration_synapses']['amin'],
                               benchmark['duration_init']['amin'],
                               benchmark['duration_run']['amin'],
                               benchmark['total']['amin']],
                              axis=1)
        benchmark.columns = ['n_neurons', 'runtime', 'duration_compile',
                             'duration_before', 'duration_after',
                             'duration_synapses', 'duration_init',
                             'duration_run', 'total']
        benchmark = benchmark.groupby(['n_neurons'])
        benchmark = benchmark.agg([np.min]).reset_index()

    # Prepare data for stacked plot
    x = np.array(sorted(benchmark['n_neurons'].unique()))
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
        runtime = np.unique(benchmark['runtime'])
        assert len(runtime) == 1
        runtime = runtime[0]
        if biological_time == runtime:
            marker = 'o'
        else:
            marker = '.'
        ax_sim.plot(np.log(x), simulate*biological_time,
                    marker=marker, mec='white', color=color_sim)
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
    ax_detail.set(xticks=np.log(ticks),
                  xlabel='Number of neurons',
                  ylabel='Wall clock time (s)')
    ax_sim.set(xlabel='', ylabel='Wall clock time (s)', yscale='log')
    ax_sim.set_yticks([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    ax_sim.spines['bottom'].set_visible(False)
    ax_detail.set_yscale('symlog', linthreshy=0.01, linscaley=np.log10(np.e))
    ax_detail.set_xticklabels(ticks, rotation=45)
    plt.setp(ax_sim.xaxis.get_ticklines(), visible=False)
    if title is not None:
        ax_sim.set_title(title)


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

    with_monitor = False

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

        # Plot details for GPU-only
        benchmark_names = ['Mbody', 'COBAHH']
        for benchmark1, benchmark2, name in [(Mbody, COBA, 'double precision'),
                                             (Mbody32, COBA32, 'single precision')]:
            gpu1 = benchmark1.loc[(benchmark1['device'] == 'genn') &
                                  (benchmark1['n_threads'] == 0)]
            gpu2 = benchmark2.loc[(benchmark2['device'] == 'genn') &
                                  (benchmark2['n_threads'] == 0)]

            fig, axes = plt.subplots(2, 2, sharey='row', sharex='col',
                                     figsize=(6.3, 6.3 * .666),
                                     gridspec_kw={'height_ratios': [1, 2]})
            plot_detailed_times(gpu1, axes[1, 0], axes[0, 0],
                                ticks=MBody_xticks,
                                legend={'simulation': (300, 1e3),
                                        'code gen & compile': (300, 20),
                                        'overhead': (300, 0.9),
                                        'synapse creation & initialization': (2700, 1.1e-2)},
                                title=benchmark_names[0] + '– ' + name)
            plot_detailed_times(gpu2, axes[1, 1], axes[0, 1],
                                ticks=COBAHH_xticks,
                                title=benchmark_names[1] + '– ' + name)
            axes[1, 1].set_ylabel(None)
            fig.tight_layout()
            fig.savefig(os.path.join(directory,
                                     'gpu_detailed_runtime_%s%s' % (name, FIGURE_EXTENSION)))
            plt.close(fig)
