# Benchmark to run ResNeXt101 on xgmi

import os
import re
import signal
import shutil
import subprocess
import time
from itertools import product

import docker

models = ['ResNeXt101_32C_48d']
num_gpus_strs = ['0,1,2,3,4,5,6,7']  # possibly add another config, '1,2,3,4'
precisions = ['FP32']
batch_sizes = ['128']
train_iterations = ['10']
experiments = ['1', '2', '3']

configs = product(models, num_gpus_strs, precisions, batch_sizes, train_iterations, experiments)

client = docker.from_env()
#                      1.7.0a0+8deb4fe'
# torch.__version__ = '1.7.0a0+fc1103a' <- output from inside container

container_img = 'rocm/pytorch:rocm3.8_ubuntu18.04_py3.6_pytorch'
container_name = 'xgmi-proof-point'

results_path = '/home/isvperf/amd-mlperf/xgmi-demo/results-11-10-2-20'
results_bind = '/data/results'

perf_path = '/home/isvperf/amd-mlperf/xgmi-demo'
perf_bind = '/data/xgmi-demo'

container_workdir = perf_bind

if not os.path.isdir(results_path):
    os.makedirs(results_path)

if not os.path.isdir(perf_path):
    raise RuntimeError('No MLPerf directory found')

if not os.path.isdir(results_path):
    os.makedirs(results_path)

def timeout_handler(signum, frame):
    raise TimeoutError('Benchmark timed out')

def timeout(func):
    def timed(*args, **kwargs):
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5400)
            return func(*args, **kwargs)
        except TimeoutError as e:
            raise e
        finally:
            signal.alarm(0)
    return timed

def check_num_mi100_gpus(n=8):
    print("Checking number of gpus...")
    call = ['/opt/rocm/bin/rocm-smi']
    output = subprocess.check_output(call)
    num_gpus = 0
    for line in output.decode('ascii').split('\n'):
        if re.match('\d(\s+\d+\.\d+[c|W]){2}(\s+\d+Mhz){2}\s+\d+\.\d+%\s+\w+\s+\d+\.\d+W(\s+\d+%){2}', line):
            num_gpus += 1

    print(f"Found {num_gpus} GPUs in the system")
    if n != num_gpus:
        raise RuntimeError(f'Incorrect number of gpus found {num_gpus}. Expected {n}')


def stop_running_container(name):
    try:
        container = client.containers.get(name)
        print("Found Container. Stopping")
        container.stop()
        time.sleep(5)
    except:
        print("No running container found")

def start_container():
    stop_running_container(container_name)
    print("Starting container")
    try:
        container = client.containers.run(
            container_img,
            volumes = {
                perf_path:
                {
                    'bind': perf_bind,
                    'mode': 'rw'
                },
                results_path:
                {
                    'bind': results_bind,
                    'mode': 'rw'
                }
            },
            detach = True,
            tty = True,
            # volumes
            shm_size = '32G',
            privileged = True,
            # cap_add = 'SYS_PTRACE',
            security_opt=['seccomp=unconfined'],
            network = 'host',
            devices = [ '/dev/kfd', '/dev/dri'],
            group_add = [ 'video' ],
            name = container_name,
            auto_remove=True
        )
    except:
        raise
    return container

def extract_throughput(lines):
    print(lines)
    for line in lines:
        if 'Throughput [img/sec]' in line:
            print(line)


@timeout
def benchmark(config, container):
    print("Running benchmark")
    model, num_gpus_str, precision, batch_size, train_steps, experiment_nb = config

    results_file = f'model_{model}_gpus_{num_gpus_str}_prec_{precision}_bs_{batch_size}_experiment_{experiment_nb}'

    cmd = f'python3.6 moded_micro_benchmarking_pytorch.py --network {model}  --dataparallel --device_ids={num_gpus_str} --iterations {train_steps} --batch-size {batch_size}'
    print(cmd)

    output = container.exec_run(cmd, workdir=container_workdir, environment=["HSA_FORCE_FINE_GRAIN_PCIE=1"])
    print(output[1])  # output[0] is the exit code 
    with open(os.path.join(results_path, results_file), 'w') as f:
        f.write(output[1].decode('ascii'))
    output_lines = [x.decode('ascii') for x in output[1].split(b'\n')]
    print(output_lines)

    extract_throughput(output_lines)


def run_benchmarks():
    print('Running benchmarks')

    for config in configs:
        try:
            print("Running config ", config)
            check_num_mi100_gpus()
            container = start_container()
            benchmark(config, container)
        except:
            raise
        finally:
            try:
                container.stop()
            except:
                pass

if __name__ == '__main__':
    run_benchmarks()





