import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('figures.conf')

directory = None  # replace this or pass the directory as a command line arg
if directory is None:
    if len(sys.argv) == 2:
        directory = sys.argv[1]
    else:
        raise ValueError('Need the directory name as an argument')


def load_benchmark(directory, fname):
    full_fname = os.path.join(directory, fname)
    benchmarks = pd.read_csv(full_fname, sep=r'\s+', header=None, index_col=None,
                             names=['device', 'n_threads', 'n_neurons', 'n_synapses',
                                    'runtime', 'with_monitor', 'total',
                                    't_after_load',
                                    't_before_synapses', 't_after_synapses',
                                    't_after_init', 't_before_run',
                                    't_after_run', 't_before_write', 't_after_write'],
                             dtype={'device': 'category',
                                    'with_monitor': 'bool',
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


def mean_and_std_fixed_time(benchmarks, monitor=True, runtime=1):
    subset = benchmarks.loc[(benchmarks['with_monitor'] == monitor) &
                            (benchmarks['runtime'] == runtime)]

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
                 16: 'violet'}[n_threads]
        return label, color


def inside_title(ax, text, x=0.98, y=.04):
    t = ax.text(x, y, text, weight='bold', horizontalalignment='right',
                     transform=ax.transAxes)
    t.set_bbox({'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8})


def plot_total(benchmarks, ax, legend=False, skip=('Brian2GeNN CPU',
                                                   'C++ 2 threads',
                                                   'C++ 16 threads')):
    ax.set_yscale('log')
    # We do the log scale for the x axis manually -- easier to get the ticks/labels right
    conditions = benchmarks.groupby(['device', 'n_threads'])
    for condition in conditions:
        (device, threads), results = condition
        label, color = label_and_color(device, threads)
        if label in skip:
            continue
        ax.plot(np.log(results['n_neurons'].values),
                results['total']['amin'],
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
           ylabel='Total wall clock time (including compilation)')


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


if __name__ == '__main__':
    for with_monitor in [True, False]:
        monitor_suffix = '' if with_monitor else '_no_monitor'
        # COBA with linear scaling
        COBA = mean_and_std_fixed_time(load_benchmark(directory, 'benchmarks_COBAHH.txt'), monitor=with_monitor)
        # Mushroom body
        Mbody = mean_and_std_fixed_time(load_benchmark(directory, 'benchmarks_Mbody_example.txt'), monitor=with_monitor)

        # Total time (including compilation)
        fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey='row',
                                                figsize=(6.3, 6.3 * .666))
        plot_total(COBA, ax_left, legend=True)
        ax_left.set_title('COBAHH')
        plot_total(Mbody, ax_right, legend=False)
        ax_right.set_title('Mushroom body')
        plt.tight_layout()
        fig.savefig(os.path.join(directory, 'total_runtime%s.pdf' % monitor_suffix))

        for benchmarks, name in [(COBA, 'COBAHH'),
                                 (Mbody, 'Mbody')]:
            fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey='row',
                                                    figsize=(6.3, 6.3 * .666))
            plot_detailed_times(benchmarks.loc[(benchmarks['device'] == 'genn') &
                                               (benchmarks['n_threads'] == 0)],
                                ax_left, legend=True)
            inside_title(ax_left, '%s (Brian2GeNN GPU)' % name)
            plot_detailed_times(benchmarks.loc[(benchmarks['device'] == 'cpp_standalone') &
                                               (benchmarks['n_threads'] == 8)],
                                ax_right)
            inside_title(ax_right, '%s (C++ 8 threads)' % name)
            plt.tight_layout()
            fig.savefig(os.path.join(directory,
                                     'detailed_runtime_%s%s.pdf' % (name, monitor_suffix)))

        for benchmarks, name in [(COBA, 'COBAHH'),
                                 (Mbody, 'Mbody')]:
            fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey='row',
                                                    figsize=(6.3, 6.3 * .666))
            plot_detailed_times_stacked(benchmarks.loc[(benchmarks['device'] == 'genn') &
                                                       (benchmarks['n_threads'] == 0)],
                                        ax_left, legend=True)
            inside_title(ax_left, '%s (Brian2GeNN GPU)' % name)
            plot_detailed_times_stacked(benchmarks.loc[(benchmarks['device'] == 'cpp_standalone') &
                                                       (benchmarks['n_threads'] == 8)],
                                        ax_right)
            inside_title(ax_right, '%s (C++ 8 threads)' % name)
            plt.tight_layout()
            fig.savefig(os.path.join(directory,
                                     'detailed_runtime_relative_%s%s.pdf' % (name, monitor_suffix)))
