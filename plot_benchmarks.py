import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('figures.conf')
directory = 'benchmark_results/2018-09-06_vuvuzela'


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
                 4: 'lightred',
                 8: 'darkred',
                 16: 'violet'}[n_threads]
        return label, color


def plot_total(benchmarks, ax, legend=False):
    ax.set_yscale('log')
    # We do the log scale for the x axis manually -- easier to get the ticks/labels right
    conditions = benchmarks.groupby(['device', 'n_threads'])
    for condition in conditions:
        (device, threads), results = condition
        label, color = label_and_color(device, threads)
        ax.plot(np.log(results['n_neurons'].values),
                results['total']['amin'],
                'o-', label=label, color=color)
    if legend:
        ax.legend(loc='upper left')
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
    Y = np.zeros((4, len(x)))
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
    for data, label in [(build, 'code generation & build'),
                        (read_write_convert, 'overhead'),
                        (create_initialize, 'synapse creation & initialization'),
                        (simulate, 'simulation')]:
        ax.plot(np.log(x), data, label=label)
    ax.plot(np.log(x), total, 'k:', label='Total')
    if legend:
        ax.legend(loc='upper left')
    # Make sure we show the xtick label for the highest value
    if len(x) % 2 == 0:
        start = 1
    else:
        start = 0
    ax.set(xticks=np.log(x)[start::2],
           xlabel='Model size (# neurons)',
           ylabel='Wall clock time (s)')
    ax.set_xticklabels(sorted(x)[start::2], rotation=45)


if __name__ == '__main__':
    # COBA with linear scaling
    COBA = mean_and_std_fixed_time(load_benchmark(directory, 'benchmarks_COBAHH.txt'))
    # Mushroom body
    Mbody = mean_and_std_fixed_time(load_benchmark(directory, 'benchmarks_Mbody_example.txt'))

    # Total time (including compilation)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey='row',
                                            figsize=(6.3, 6.3 * .666))
    plot_total(COBA, ax_left, legend=True)
    ax_left.set_title('COBAHH')
    plot_total(Mbody, ax_right, legend=False)
    ax_right.set_title('Mushroom body')

    fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey='row',
                                            figsize=(6.3, 6.3 * .666))
    plot_detailed_times(COBA.loc[(COBA['device'] == 'genn') & (COBA['n_threads'] == 0)],
                        ax_left, legend=True)
    ax_left.set_title('COBAHH -- Brian2GeNN GPU')
    plot_detailed_times(COBA.loc[(COBA['device'] == 'cpp_standalone') & (COBA['n_threads'] == 8)],
                        ax_right)
    ax_right.set_title('COBAHH -- C++ 8 threads')

    fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey='row',
                                            figsize=(6.3, 6.3 * .666))
    plot_detailed_times(Mbody.loc[(Mbody['device'] == 'genn') & (Mbody['n_threads'] == 0)],
                        ax_left, legend=True)
    ax_left.set_title('Mbody -- Brian2GeNN GPU')
    plot_detailed_times(Mbody.loc[(Mbody['device'] == 'cpp_standalone') & (Mbody['n_threads'] == 8)],
                        ax_right)
    ax_right.set_title('Mbody -- C++ 8 threads')

    plt.show()
