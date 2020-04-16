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
directory= 'benchmark_results/2020-04-15_inf900777/'
directory_old = 'benchmark_results/2018-10-02_inf900777/'  # replace this or pass the directory as a command line arg
algo= 'post'

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

def plot_detailed_times(benchmark, bench_old, ax_detail, ax_sim, ticks, title=None, legend=False, algo=None):
    if (algo == None):
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
            benchmark = benchmark.groupby(['n_neurons','runtime'])
            benchmark = benchmark.agg([np.min]).reset_index()
            if len(bench_old['algorithm'].unique()) > 1:
                bench_old = pd.concat([bench_old['n_neurons'],
                                       bench_old['runtime'],
                                       bench_old['duration_compile']['amin'],
                                       bench_old['duration_before']['amin'],
                                       bench_old['duration_after']['amin'],
                                       bench_old['duration_synapses']['amin'],
                                       bench_old['duration_init']['amin'],
                                       bench_old['duration_run']['amin'],
                                       bench_old['total']['amin']],
                                      axis=1)
                bench_old.columns = ['n_neurons', 'runtime', 'duration_compile',
                                     'duration_before', 'duration_after',
                                     'duration_synapses', 'duration_init',
                                     'duration_run', 'total']
                bench_old = bench_old.groupby(['n_neurons','runtime'])
            bench_old = bench_old.agg([np.min]).reset_index()
            bench_old = bench_old.loc[bench_old['duration_run'].notnull()['amin']]
    else:
        benchmark = benchmark.loc[benchmark['algorithm'] == algo]
        bench_old = bench_old.loc[bench_old['algorithm'] == algo]
    bench_old = bench_old.loc[bench_old['duration_run'].notnull()['amin']]
    benchmark = benchmark.loc[benchmark['duration_run'].notnull()['amin']]
      
    # Prepare data for stacked plot
    x = np.array(sorted(benchmark['n_neurons'].unique()))
    x_old = np.array(sorted(bench_old['n_neurons'].unique()))
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
   # Prepare data for stacked plot
    # code generation and build
    build_old = bench_old['duration_compile']['amin']
    # Reading/writing arrays from/to disk / conversion between Brian2 and GeNN format
    read_write_convert_old = (bench_old['duration_before']['amin'] +
                          bench_old['duration_after']['amin'])
    # Creating synapses and initialization
    create_initialize_old = (bench_old['duration_synapses']['amin'] +
                         bench_old['duration_init']['amin'])
    # Simulation time
    simulate_old = bench_old['duration_run']['amin']
    # TODO: Check that the total matches
    total_old = bench_old['total']['amin']
    handles = []
    labels = []
    for tx, data, label in [(x, build, 'code gen & compile'),
                        (x, read_write_convert, 'overhead'),
                        (x, create_initialize, 'synapse creation & initialization'),
                        (x_old, build_old, 'code gen & compile'),
                        (x_old, read_write_convert_old, 'overhead'),
                        (x_old, create_initialize_old, 'synapse creation & initialization')]:
        h = ax_detail.plot(np.log(tx), data, 'o-', mec='white')[0]
        
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
        basecolor_sim_old= colorsys.rgb_to_hls(*mpl.cm.tab10.colors[4])
        color_sim = colorsys.hls_to_rgb(basecolor_sim[0],
                                        basecolor_sim[1] + brighten_by*(1 - basecolor_sim[1]),
                                        basecolor_sim[2])
        color_sim_old = colorsys.hls_to_rgb(basecolor_sim_old[0],
                                        basecolor_sim_old[1] + brighten_by*(1 - basecolor_sim_old[1]),
                                        basecolor_sim_old[2])
        runtime = benchmark['runtime']
        runtime_old= bench_old['runtime']
        marker= '.'
        marker_old= '+'
        ax_sim.plot(np.log(x_old), simulate_old/runtime_old*biological_time,
                     marker=marker_old,  color=color_sim_old, linewidth= 1)
        ax_sim.plot(np.log(x), simulate/runtime.to_numpy().flatten()*biological_time,
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
    ax_sim.set(xticks=np.log(ticks))
    ax_sim.set(xlabel='', ylabel='Wall clock time (s)', yscale='log')
    ax_sim.set_yticks([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    ax_sim.spines['bottom'].set_visible(False)
    ax_detail.set_yscale('symlog', linthreshy=0.01, linscaley=np.log10(np.e))
    ax_detail.set_xticklabels(['{:,d}'.format(t) for t in ticks], rotation=45)
    plt.setp(ax_sim.xaxis.get_ticklines(), visible=False)
    if title is not None:
        ax_sim.set_title(title)


def plot_times_compare(benchmark, bench_old, ax_detail, ax_sim, ticks, title=None, legend=False, algo=None):
    if (algo == None):
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
            benchmark = benchmark.groupby(['n_neurons','runtime'])
            benchmark = benchmark.agg([np.min]).reset_index()
            if len(bench_old['algorithm'].unique()) > 1:
                bench_old = pd.concat([bench_old['n_neurons'],
                                       bench_old['runtime'],
                                       bench_old['duration_compile']['amin'],
                                       bench_old['duration_before']['amin'],
                                       bench_old['duration_after']['amin'],
                                       bench_old['duration_synapses']['amin'],
                                       bench_old['duration_init']['amin'],
                                       bench_old['duration_run']['amin'],
                                       bench_old['total']['amin']],
                                      axis=1)
                bench_old.columns = ['n_neurons', 'runtime', 'duration_compile',
                                     'duration_before', 'duration_after',
                                     'duration_synapses', 'duration_init',
                                     'duration_run', 'total']
                bench_old = bench_old.groupby(['n_neurons','runtime'])
            bench_old = bench_old.agg([np.min]).reset_index()
            bench_old = bench_old.loc[bench_old['duration_run'].notnull()['amin']]
    else:
        benchmark = benchmark.loc[benchmark['algorithm'] == algo]
        bench_old = bench_old.loc[bench_old['algorithm'] == algo]
    bench_old = bench_old.loc[bench_old['duration_run'].notnull()['amin']]
    benchmark = benchmark.loc[benchmark['duration_run'].notnull()['amin']]
      
    # Prepare data for stacked plot
    x = np.array(sorted(benchmark['n_neurons'].unique()))
    x_old = np.array(sorted(bench_old['n_neurons'].unique()))
    
    sz= min(len(x),len(x_old))
    tx= x[:sz]
    # code generation and build
    build = benchmark['duration_compile']['amin']
    print(build)
    build = build[:sz]
    print(build)
    print(sz)
    # Reading/writing arrays from/to disk / conversion between Brian2 and GeNN format
    read_write_convert = (benchmark['duration_before']['amin'] +
                          benchmark['duration_after']['amin'])
    read_write_convert = read_write_convert[:sz]
    # Creating synapses and initialization
    create_initialize = (benchmark['duration_synapses']['amin'] +
                         benchmark['duration_init']['amin'])
    create_initialize = create_initialize [:sz]
    # Simulation time
    simulate = benchmark['duration_run']['amin']
    simulate = simulate[:sz]
   # Prepare data for stacked plot
    # code generation and build
    build_old = bench_old['duration_compile']['amin']
    build_old= build_old[:sz]
    # Reading/writing arrays from/to disk / conversion between Brian2 and GeNN format
    read_write_convert_old = (bench_old['duration_before']['amin'] +
                          bench_old['duration_after']['amin'])
    read_write_convert_old= read_write_convert_old[:sz]
    # Creating synapses and initialization
    create_initialize_old = (bench_old['duration_synapses']['amin'] +
                         bench_old['duration_init']['amin'])
    create_initialize_old= create_initialize_old[:sz]
    # Simulation time
    simulate_old = bench_old['duration_run']['amin']
    simulate_old = simulate_old[:sz]
    handles = []
    labels = []
    for data, label in [(build_old.to_numpy()/build.to_numpy(), 'code gen & compile'),
                        (read_write_convert_old.to_numpy()/read_write_convert.to_numpy(), 'overhead'),
                        (create_initialize_old.to_numpy()/create_initialize.to_numpy(), 'synapse creation & initialization')]:
        print(tx)
        print(data)
        h = ax_detail.plot(np.log(tx), data, 'o-', mec='white')[0]
        
        if legend:
            xl, yl = legend[label]
            ax_detail.text(np.log(xl), yl, label,
                           fontsize=8, fontweight='bold', color=h.get_color())
    ax_detail.grid(b=True, which='major', color='#b0b0b0', linestyle='-',
                   linewidth=0.5)
    # We plot simulation/total times for 1s, 10s, and 100s
    import colorsys
    color_sim = colorsys.rgb_to_hls(*mpl.cm.tab10.colors[3])
    color_sim_old= colorsys.rgb_to_hls(*mpl.cm.tab10.colors[4])
    runtime = benchmark['runtime']
    runtime= runtime.to_numpy()[:sz]
    simulate= simulate.to_numpy()[:sz]
    simulate= simulate.flatten()/runtime.flatten()
    runtime_old= bench_old['runtime']
    runtime_old= runtime_old.to_numpy()[:sz]
    simulate_old= simulate_old.to_numpy()[:sz]
    simulate_old= simulate_old.flatten()/runtime_old.flatten()
    marker= '.'
    marker_old= '+'
    print(simulate/simulate_old)
    ax_sim.plot(np.log(tx), simulate_old/simulate,
                marker=marker_old,  color=color_sim, linewidth= 1)
    if legend:
        xl, yl = legend['simulation']
        ax_sim.text(np.log(xl), yl, 'simulation',
                    fontsize=8, color=color_sim,
                    fontweight='bold')
    ax_sim.grid(b=True, which='major', color='#b0b0b0', linestyle='-',
                linewidth=0.5)
    ax_detail.set(xticks=np.log(ticks),
                  xlabel='Number of neurons',
                  ylabel='speedup (unitless)')
    ax_sim.set(xticks=np.log(ticks))
    ax_sim.set(xlabel='', ylabel='speedup (unitless')
    #ax_sim.set_yticks([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    ax_sim.spines['bottom'].set_visible(False)
    #ax_detail.set_yscale('symlog', linthreshy=0.01, linscaley=np.log10(np.e))
    ax_sim.set(ylim= [ 0, 7 ])
    ax_detail.set_xticklabels(['{:,d}'.format(t) for t in ticks], rotation=45)
    plt.setp(ax_sim.xaxis.get_ticklines(), visible=False)
    if title is not None:
        ax_sim.set_title(title)
        

if __name__ == '__main__':
    if directory is None:
        if len(sys.argv) == 2:
            directory = sys.argv[1]
        else:
            raise ValueError('Need the directory name as an argument')

    if algo == None:
        algo_str= ''
    else:
        algo_str= algo

    COBA_full = load_benchmark(directory, 'benchmarks_COBAHH.txt')
    Mbody_full = load_benchmark(directory, 'benchmarks_Mbody_example.txt')
    COBA_full_old = load_benchmark(directory_old, 'benchmarks_COBAHH.txt')
    Mbody_full_old = load_benchmark(directory_old, 'benchmarks_Mbody_example.txt')

    COBA32 = mean_and_std_fixed_time(COBA_full, monitor=False, float_dtype='float32')
    COBA64 = mean_and_std_fixed_time(COBA_full, monitor=False, float_dtype='float64')
    Mbody32 = mean_and_std_fixed_time(Mbody_full, monitor=False, float_dtype='float32')
    Mbody64 = mean_and_std_fixed_time(Mbody_full, monitor=False, float_dtype='float64')
    COBA32_old = mean_and_std_fixed_time(COBA_full_old, monitor=False, float_dtype='float32')
    COBA64_old = mean_and_std_fixed_time(COBA_full_old, monitor=False, float_dtype='float64')
    Mbody32_old = mean_and_std_fixed_time(Mbody_full_old, monitor=False, float_dtype='float32')
    Mbody64_old = mean_and_std_fixed_time(Mbody_full_old, monitor=False, float_dtype='float64')
    labels = ['COBA – single prec', 'COBA – double prec',
              'Mbody – single prec', 'Mbody – double prec']
    # COBA with linear scaling
    COBA64_MON = mean_and_std_fixed_time(COBA_full, monitor=True, float_dtype='float64')
    COBA32_MON = mean_and_std_fixed_time(COBA_full, monitor=True,float_dtype='float32')
    COBA64_MON_old = mean_and_std_fixed_time(COBA_full_old, monitor=True, float_dtype='float64')
    COBA32_MON_old = mean_and_std_fixed_time(COBA_full_old, monitor=True, float_dtype='float32')
    # Mushroom body
    Mbody64_MON = mean_and_std_fixed_time(Mbody_full, monitor=True, float_dtype='float64')
    Mbody32_MON = mean_and_std_fixed_time(Mbody_full, monitor=True,float_dtype='float32')
    Mbody64_MON_old = mean_and_std_fixed_time(Mbody_full_old, monitor=True, float_dtype='float64')
    Mbody32_MON_old = mean_and_std_fixed_time(Mbody_full_old, monitor=True, float_dtype='float32')

    # Plot details for GPU-only
    benchmark_names = ['Mbody', 'COBAHH']
    for benchmark1, benchmark2, bench1_old, bench2_old, name in [(Mbody64, COBA64, Mbody64_old, COBA64_old, 'double prec no mon'),
                                                                 (Mbody32, COBA32, Mbody32_old, COBA32_old, 'single prec no mon')
                                                                 ]:
        gpu1 = benchmark1.loc[(benchmark1['device'] == 'genn') &
                              (benchmark1['n_threads'] == 0)]
        gpu2 = benchmark2.loc[(benchmark2['device'] == 'genn') &
                              (benchmark2['n_threads'] == 0)]

        gpu1_old = bench1_old.loc[(bench1_old['device'] == 'genn') &
                              (bench1_old['n_threads'] == 0)]
        gpu2_old = bench2_old.loc[(bench2_old['device'] == 'genn') &
                              (bench2_old['n_threads'] == 0)]
        fig, axes = plt.subplots(2, 2, sharey='row', sharex='col',
                                 figsize=(6.3, 6.3 * .666),
                                 gridspec_kw={'height_ratios': [1, 2]})
        plot_detailed_times(gpu1, gpu1_old, axes[1, 0], axes[0, 0],
                            ticks=MBody_xticks,
                            legend={'simulation': (300, 1e3),
                                    'code gen & compile': (300, 20),
                                    'overhead': (300, 0.9),
                                    'synapse creation & initialization': (2700, 1.1e-2)},
                            title=benchmark_names[0] + '– ' + name+' '+algo_str ,
                            algo=algo)
        plot_detailed_times(gpu2, gpu2_old, axes[1, 1], axes[0, 1],
                            ticks=COBAHH_xticks,
                            title=benchmark_names[1] + '– ' + name +' ' + algo_str,
                            algo=algo)
        axes[0, 1].set_ylabel(None)
        axes[1, 1].set_ylabel(None)
        fig.tight_layout()
        fig.savefig(os.path.join(directory,
                                 'gpu_detailed_runtime_%s%s%s' % (name, '_'+algo_str, FIGURE_EXTENSION)))
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, sharey='row', sharex='col',
                                 figsize=(6.3, 6.3 * .666),
                                 gridspec_kw={'height_ratios': [1, 2]})

        plot_times_compare(gpu1, gpu1_old, axes[1, 0], axes[0, 0],
                            ticks=MBody_xticks,
                            legend={'simulation': (300, 2),
                                    'code gen & compile': (5200, 1),
                                    'overhead': (150000, 0.7),
                                    'synapse creation & initialization': (300, 0.2)},
                            title=benchmark_names[0] + '– ' + name+' '+algo_str ,
                            algo=algo)
        plot_times_compare(gpu2, gpu2_old, axes[1, 1], axes[0, 1],
                            ticks=COBAHH_xticks,
                            title=benchmark_names[1] + '– ' + name +' ' + algo_str,
                            algo=algo)
        axes[0, 1].set_ylabel(None)
        axes[1, 1].set_ylabel(None)
        fig.tight_layout()
        fig.savefig(os.path.join(directory,
                                 'gpu_detailed_runtime_compare_%s%s%s' % (name, '_'+algo_str, FIGURE_EXTENSION)))
        plt.close(fig)
