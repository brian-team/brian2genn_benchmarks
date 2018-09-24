import warnings
import os

from brian2 import *

TIME_DIFF = 'std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - _benchmark_start).count()'
NOTE_TIME = '_benchmark_file << {timediff} << " ";'.format(timediff=TIME_DIFF)


def prepare_benchmark(argv):
    config = {}
    extra_args = {}

    if len(argv) <= 2:  # No or single argument --> debugmode
        if len(argv) == 1:
            config['scale'] = 1.0
        else:
            config['scale'] = float(sys.argv[1])
        config['device'] = 'cpp_standalone'
        config['threads'] = 1
        config['runtime'] = 1.0*second
        config['monitor'] = True
        config['float_dtype'] = 'float64'
        config['debug'] = True
    else:
        config['scale'] = float(sys.argv[1])
        config['device'] = sys.argv[2]
        config['threads'] = int(sys.argv[3])
        config['runtime'] = float(sys.argv[4])*second
        config['monitor'] = sys.argv[5].lower() == 'true' or sys.argv[5] == '1'
        config['float_dtype'] = sys.argv[6]
        config['label'] = sys.argv[7]
        config['debug'] = False

    if config['device'] == 'genn':
        prefs.devices.genn.auto_choose_device = False
        prefs.devices.genn.default_device = 0
        prefs.devices.genn.synapse_span_type = 'PRESYNAPTIC'
        if config['threads'] > 0:
            warnings.warn('Brian2GeNN cannot run multi-threaded, setting '
                          'number of threads to 0')
            config['threads'] = 0
    else:
        extra_args['build_on_run'] = False

    if config['threads'] == -1:
        assert config['device'] == 'genn'
        extra_args = {'use_GPU': False}
    else:
        prefs.devices.cpp_standalone.openmp_threads = config['threads']

    prefs.codegen.cpp.extra_compile_args_gcc += ['-std=c++11']
    prefs.codegen.cpp.extra_compile_args_msvc += ['/std:c++14']
    prefs.codegen.cpp.headers += ['<chrono>']
    if config['float_dtype'] == 'float32':
        prefs.core.default_float_dtype = np.float32
    else:
        prefs.core.default_float_dtype = np.float64
    set_device(config['device'], **extra_args)
    return config


def insert_general_benchmark_code(config):
    directory = 'GeNNworkspace' if config['device'] == 'genn' else 'output'
    fname = os.path.abspath(os.path.join(directory, 'results',
                                         'benchmark.time'))
    device.insert_code('before_start',
                       '''std::chrono::high_resolution_clock::time_point _benchmark_start = std::chrono::high_resolution_clock::now();
std::ofstream _benchmark_file;
_benchmark_file.open("{fname}");'''.format(fname=fname))
    device.insert_code('after_start', NOTE_TIME)
    device.insert_code('before_end', NOTE_TIME)
    device.insert_code('after_end', NOTE_TIME + '\n_benchmark_file.close();')


def insert_benchmark_point():
    device.insert_code('main', NOTE_TIME)


def do_and_measure_run(config):
    import time

    start = time.time()
    if config['device'] == 'cpp_standalone':
        device.insert_code('main', NOTE_TIME)
        run(config['runtime'], report='text', level=1)
        device.insert_code('main', NOTE_TIME)
        device.build()
    else:
        device.insert_code('before_run', NOTE_TIME)
        device.insert_code('after_run', NOTE_TIME)
        run(config['runtime'], report='text', level=1)

    took = (time.time() - start)
    print('Took %.1fs in total (runtime: %.1fs)' % (took, device._last_run_time))
    return took


def write_benchmark_results(name, config, neurons, synapses, took):
    directory = 'GeNNworkspace' if config['device'] == 'genn' else 'output'
    # Note that we do not use the device._last_run_time, we'll use the
    # difference between the times just before and after the run instead
    with_monitor = int(config['monitor'])
    folder = os.path.join('benchmark_results', config['label'])
    if not os.path.exists(folder):
        os.mkdir(folder)
    with open('%s/benchmarks_%s.txt' % (folder, name), 'a') as f:
        data = [config['device'], config['threads'], neurons, synapses,
                config['runtime'] / second, with_monitor,
                prefs.core.default_float_dtype.__name__, took]
        f.write('\t'.join('%s' % d for d in data) + '\t')
        with open('%s/results/benchmark.time' % directory, 'r') as bf:
            for line in bf:
                line = line.strip()
                line = '\t'.join(
                    '%s' % item for item in line.split(' ')) + '\n'
                f.write(line)
