# Benchmarking for Brian2 and Brian2GeNN


These files can be used to run detailed benchmarking on Brian2's C++ standalone
mode and on Brian2GeNN. It is meant to run with the most recent versions of
Brian2/Brian2GeNN/GeNN, i.e. currently the master branch (all projects will soon
have corresponding releases: 2.2 (Brian), 1.2 (Brian2GeNN), and 3.2 (GeNN)).

## Benchmarks
Currently, there are two benchmarks:
* `COBAHH.py`, a recurrent network of Hodgkin-Huxley type neurons with 
   conductance-based synapses, basically the same model as the
   [example in the Brian 2 documentation](https://brian2.readthedocs.io/en/stable/examples/COBAHH.html).
   This example has been slightly changed to make scaling its size easier:
   neurons connect on average with 1000 synapses to other neurons (this means 
   that the number of synapses scales linearly with the number of neurons), but
   the synaptic weights are set to 0. The neurons still spike because of their
   constant input currents, and the simulation calculates synaptic updates, but
   we don't have to worry about a change in firing rates when scaling the size
   of the network.
* `Mbody_example`, a model of the locust olfactory system (Nowotny et al. 2005),
   with several different synapse types including synapses with
   spike-timing-dependent plasticity. This models scales the number of neurons
   in the mushroom body only, and keeps the number of plastic synapses in the
   mushroom body at 10000 per neuron (for networks with > 10000 neurons).

Each of these examples can be run manually by running it with certain
command-line arguments:
```console
$ python [benchmark] [scale] [device] [n_threads] [runtime] [monitor] [float_dtype] [label]
``` 
where:
<dl>
<dt>benchmark</dt>
<dd>the name of the benchmark file</dd>
<dt>scale</dt>
<dd>The scaling factor to use for the simulation (1 means the "usual" number of 
neurons, 2 twice this number, etc.)</dd>
<dt>device</dt>
<dd>Either <code>genn</code> or <code>cpp_standalone</code></dd>
<dt>n_threads</dt>
<dd>The number of threads to use in C++ standalone mode. For Brian2GeNN, either
-1 to run the simulation on the CPU, or 0 to run it on the GPU</dd>
<dt>runtime</dt>
<dd>The biological runtime of the simulation in seconds</dd>
<dt>monitor</dt>
<dd>Whether to use monitors in the simulation (spike monitors for both
benchmarks, additional a monitor recording the membrane potential of 1% of the
neurons in the COBAHH benchmark)</dd>
<dt>float_dtype</dt>
<dd>The datatype for floating point variables, either <code>float32</code> for
single precision or <code>float64</code> for double precision.</dd>
<dt>label</dt>
<dd>A label that will be used to decide in which directory to put the benchmark
(i.e., use the same label for benchmarks run on the same machine)</dd>
</dl>

The provided bash files can be used to run benchmarks for combinations of these
arguments (these files are meant for use on Linux or OS X, but should be easily
adaptable for Windows).

## Benchmarking mechanism

Benchmarking is performed by using Brian2's [`insert_code`](https://brian2.readthedocs.io/en/stable/reference/brian2.devices.device.Device.html#brian2.devices.device.Device.insert_code)
mechanism. This injected code will use a C++ `high_resolution_clock` and write
the time elapsed since the start at various points to a
`results/benchmark.time` file in the model's code directory (`output` for C++ 
standalone, `GeNNworkspace` for Brian2GeNN). To calculate the time spend in
Brian2's Python code (code generation etc.) as well as for the model
compilation, the total time is also measured in the benchmark script itself via
Python -- the difference between this measured time and the time spend executing
the generated code gives the time spent for all preparatory work.

Brian2's C++ standalone code and the code generated by Brian2GeNN are slightly
different in the way they enchain the various operations. The benchmarking
script takes care of creating comparable time points.

The following sketch of a simulation's `main` function shows the time points at
which measurements are taken, together with the names used for these points in
the analysis script (`plot_benchmarks.py`):
```C++
int main(int argc, char **argv)
{
    // <-- *** Start timer for benchmarking
    brian_start() // reserve memory for arrays and initialize them to 0
                  // load values provided in Python code from disk
    // <-- ***  time point: t_after_load
    // housekeeping code: set number of threads, initialize some variables to
    //                    scalar values != 0
    // <-- ***  time point: t_before_synapses
    // create synapses
    _run_synapses_synapses_create_generator_codeobject();
    // ...
    // <-- ***  time point: t_after_synapses
    // initialize state variables with user-provided expressions (e.g. randomly)
    _run_neurongroup_group_variable_set_conditional_codeobject(); 
    // ...
    // <-- ***  time point: t_after_init
    // more housekeeping, copying over previously loaded arrays
    // <-- ***  time point: t_before_run
    // Setting up simulation (scheduling, etc.)
    // Run the network:
    magicnetwork.run(1.0, report_progress, 10.0);
    // <-- ***  time point: t_after_run
    // <-- ***  time point: t_before_end
    brian_end();  // Write results to disk, free memory
    // <-- ***  time point: t_after_end
}
```

The generated code for Brian2GeNN is very similar:
```C++
int main(int argc, char **argv)
{
    // <-- *** Start timer for benchmarking
    _init_arrays(); // reserve memory for arrays and initialize them to 0
    _load_arrays(); // load values provided in Python code from disk
    // <-- ***  time point: t_after_load
    // initialize some variables to scalar values != 0
    // <-- ***  time point: t_before_synapses
    // create synapses
    _run_synapses_synapses_create_generator_codeobject();
    // ...
    // <-- ***  time point: t_after_synapses
    // initialize state variables with user-provided expressions (e.g. randomly)
    _run_neurongroup_group_variable_set_conditional_codeobject(); 
    // ...
    // <-- ***  time point: t_after_init
    // housekeeping: copying over previously loaded arrays
    //               convert variables from Brian 2 to GeNN format
    //               copy variables to GPU
    // <-- ***  time point: t_before_run
    // run the simulation:
    eng.run(totalTime, which);
    // <-- ***  time point: t_after_run
    // housekeeping: copy over data from GPU
    //               convert variables from Brian2 to GeNN format
    // <-- ***  time point: t_before_end
    _write_arrays();    // Write results to disk 
    _dealloc_arrays();  // free memory
    // <-- ***  time point: t_after_end
```    

The main difference between the two codes is that Brian2GeNN does more
"housekeeping", in particular it has to convert array data structures between
Brian2's and GeNN's format, and copy things back and forth between CPU and GPU.

## Benchmark evaluation
The data from the time points is summarized in the following measurements, again
using the names given in `plot_benchmarks.py`:

<dl>
<dt><code>duration_before</code></dt>
<dd>General preparation time, excluding synapse creation and variable initialization
(time between start and <code>t_after_load</code> + time between
<code>t_after_init</code> and <code>t_before_run</code>)</dd>
<dt><code>duration_synapses</code></dt>
<dd>Synapse creation time
(time between <code>t_before_synapses</code> and <code>t_after_synapses</code>)</dd>
<dt><code>duration_init</code></dt>
<dd>Variable initialization time
(time between <code>t_after_synapses</code> and <code>t_after_init</code>)</dd>
<dt><code>duration_run</code></dt>
<dd>Simulation time
(time between <code>t_before_run</code> and <code>t_after_run</code>)</dd>
<dt><code>duration_init</code></dt>
<dd>Variable initialization time
(time between <code>t_after_synapses</code> and <code>t_after_init</code>)</dd>
<dt><code>duration_after</code></dt>
<dd>Cleanup time
(time between <code>t_after_run</code> and <code>t_after_write</code>)</dd>
<dt><code>duration_compile</code></dt>
<dd>Code generation and compilation time
(difference between the total time measured in Python, and the total time
measured within the generated code, i.e.  <code>t_after_write</code>)</dd>
</dl>

In the plots, `duration_before` and `duration_after` are summed and called
"overhead", in the same way `duration_synapses` and `duration_init` are summed
to give the total time for "synapse creation & initialization". "Simulation"
directly to corresponds to `duration_run`, and "code gen & compile" corresponds
to `duration_compile`.

All measured times are the minimum times across repeats (but variation between
runs is very small).