import os
import sys
import itertools

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FIGURE_EXTENSION = '.png'

plt.style.use('figures.conf')

directory = None  # replace this or pass the directory as a command line arg
if directory is None:
    if len(sys.argv) == 2:
        directory = sys.argv[1]
        compareplot = False
    else:
        if len(sys.argv) == 3:
            directory = sys.argv[1]
            directory2 = sys.argv[2]
            compareplot= True
        else:
            raise ValueError('Need the directory name as an argument')


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


def label_and_color(device, n_threads):
    if device == 'genn':
        if n_threads == -1:
            return 'Brian2GeNN CPU', 'lightblue'
        else:
            return 'Brian2GeNN GPU', 'darkblue'
    else:
        label = 'C++ %2d threads' % n_threads
        # Quick&dirty colour selection
        color = {1: 'gold',
                 2: 'darkorange',
                 4: 'tomato',
                 8: 'darkred',
                 12: 'darkviolet',
                 16: 'indigo',
                 24: 'green',}[n_threads]
        return label, color


def inside_title(ax, text, x=0.98, y=.04):
    t = ax.text(x, y, text, weight='bold', horizontalalignment='right',
                     transform=ax.transAxes)
    t.set_bbox({'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8})


def plot_total(benchmarks, ax, legend=False, skip=('Brian2GeNN CPU',),
               plot_what='total',
               axis_label='Total wall clock time (including compilation)'):
    ax.set_yscale('log')
    # We do the log scale for the x axis manually -- easier to get the ticks/labels right
    conditions = benchmarks.groupby(['device', 'n_threads'])
    for condition in conditions:
        (device, threads), results = condition
        label, color = label_and_color(device, threads)
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


def plot_detailed_times(benchmark, ax, legend=False):
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
    for data, label in [(build, 'code gen & compile'),
                        (read_write_convert, 'overhead'),
                        (create_initialize, 'synapse creation & initialization'),
                        (simulate, 'simulation')]:
        ax.plot(np.log(x), data, 'o-', label=label, mec='white')
    ax.plot(np.log(x), total, 'ko-', mec='white', label='Total')
    ax.grid(b=True, which='minor', color='#b0b0b0', linestyle='-',
            linewidth=0.33)
    if legend:
        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                   mode="expand", borderaxespad=0, ncol=2)
        # ax.legend(loc='upper left')
    # Make sure we show the xtick label for the highest value
    if len(x) % 2 == 0:
        start = 1
    else:
        start = 0
    ax.set(xticks=np.log(x)[start::2],
           xlabel='Model size (# neurons)',
           ylabel='Wall clock time (s)',
           yscale='log')
    ax.set_xticklabels(sorted(x)[start::2], rotation=45)


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

if __name__ == '__main__':
    if not compareplot:
        COBA_full = load_benchmark(directory, 'benchmarks_COBAHH.txt')
        Mbody_full = load_benchmark(directory, 'benchmarks_Mbody_example.txt')
        
        for with_monitor in [True, False]:
            monitor_suffix = '' if with_monitor else '_no_monitor'
            # COBA with linear scaling
            COBA = mean_and_std_fixed_time(COBA_full, monitor=with_monitor)
            COBA32 = mean_and_std_fixed_time(COBA_full, monitor=with_monitor,
                                             float_dtype='float32')
            # Mushroom body
            Mbody = mean_and_std_fixed_time(Mbody_full, monitor=False)
            Mbody32 = mean_and_std_fixed_time(Mbody_full, monitor=False,
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
                        print(benchmarks)
                        max_threads = benchmarks.loc[benchmarks['device'] == 'cpp_standalone']['n_threads'].max()
                        
                        fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey='row',
                                                                figsize=(6.3, 6.3 * .666))
                        GeNN_GPU = benchmarks.loc[(benchmarks['device'] == 'genn') &
                                                  (benchmarks['n_threads'] == 0)]
                        if len(GeNN_GPU):
                            plot_detailed_times(GeNN_GPU, ax_left, legend=True)
                            inside_title(ax_left, '%s (Brian2GeNN GPU)' % name)
                        else:
                            ax_left.axis('off')
                            cpp_threads = benchmarks.loc[(benchmarks['device'] == 'cpp_standalone') &
                                                         (benchmarks['n_threads'] == max_threads)]
                            if len(cpp_threads):
                                plot_detailed_times(cpp_threads, ax_right)
                                inside_title(ax_right, '%s (C++ %d threads)' % (name, max_threads))
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
    else: #compareplot
        COBA_full = load_benchmark(directory, 'benchmarks_COBAHH.txt')
        Mbody_full = load_benchmark(directory, 'benchmarks_Mbody_example.txt')
        COBA_full_cmp = load_benchmark(directory2, 'benchmarks_COBAHH.txt')
        Mbody_full_cmp = load_benchmark(directory2, 'benchmarks_Mbody_example.txt')

        # COBA with linear scaling
        COBA = mean_and_std_fixed_time(COBA_full, monitor=True)
        COBA32 = mean_and_std_fixed_time(COBA_full, monitor=True,
                                         float_dtype='float32')
        COBA_cmp = mean_and_std_fixed_time(COBA_full_cmp, monitor=True)
        COBA32_cmp = mean_and_std_fixed_time(COBA_full_cmp, monitor=True,
                                         float_dtype='float32')
        # Mushroom body
        Mbody = mean_and_std_fixed_time(Mbody_full, monitor=False)
        Mbody32 = mean_and_std_fixed_time(Mbody_full, monitor=False,
                                              float_dtype='float32')
        Mbody_cmp = mean_and_std_fixed_time(Mbody_full_cmp, monitor=False)
        Mbody32_cmp = mean_and_std_fixed_time(Mbody_full_cmp, monitor=False,
                                              float_dtype='float32')

        for COBA_benchmark, Mbody_benchmark, COBA_b2, Mbody_b2, suffix in [(COBA, Mbody, COBA_cmp, Mbody_cmp, ' cmp'),
                                                            (COBA32, Mbody32, COBA32_cmp, Mbody32_cmp, ' cmp single precision')]:
                # Total time (including compilation)
                fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey='row',
                                                        figsize=(6.3, 6.3 * .666))
                plot_total(COBA_benchmark, ax_left, legend=True)
                ax_left.set_title('COBAHH%s' % suffix)
                plot_total(COBA_b2, ax_left, legend=True)
                ax_left.set_title('COBAHH pre %s' % suffix)
                plot_total(Mbody_benchmark, ax_right, legend=False)
                ax_right.set_title('Mushroom body%s' % suffix)
                plot_total(Mbody_b2, ax_right, legend=False)
                ax_right.set_title('Mushroom body pre %s' % suffix)
                plt.tight_layout()
                fig.savefig(os.path.join(directory, 'total_runtime%s%s' % (suffix, FIGURE_EXTENSION)))
                
                # Runtime relative to realtime
                fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey='row',
                                                        figsize=(6.3, 6.3 * .666))
                plot_total(COBA_benchmark, ax_left, legend=True, plot_what='duration_run_rel',
                           axis_label='Simulation time (relative to real-time)')
                plot_total(COBA_b2, ax_left, legend=True, plot_what='duration_run_rel',
                           axis_label='Simulation time (relative to real-time)')
                ax_left.set_title('COBAHH')
                plot_total(Mbody_benchmark, ax_right, legend=False, plot_what='duration_run_rel',
                           axis_label='Simulation time (relative to real-time)')
                plot_total(Mbody_b2, ax_right, legend=False, plot_what='duration_run_rel',
                           axis_label='Simulation time (relative to real-time)')
                ax_right.set_title('Mushroom body')
                plt.tight_layout()
                fig.savefig(os.path.join(directory, 'simulation_time_only%s%s' % (suffix, FIGURE_EXTENSION)))
