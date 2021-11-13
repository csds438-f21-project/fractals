
import time
import sys
import numpy as np

NUM_TRIALS = 10

def run_experiment(framework, num_processes):
    for N in num_processes:
        print(f'N = {N}')
        times = []
        for trial in range(NUM_TRIALS):
            if framework == 'ray':
                import ray
                from ray.util.multiprocessing import Pool
                ray.shutdown()
                ray.init()

                start_time = time.time()
                with Pool(N) as pool:
                    pass
                end_time = time.time()
            elif framework == 'lewicki':
                from lewicki import ActorPool

                start_time = time.time()
                pool = ActorPool(N)
                end_time = time.time()
            elif framework == 'mp':
                from multiprocessing import Pool

                start_time = time.time()
                with Pool(N) as pool:
                    pass
                end_time = time.time()

            times.append(end_time - start_time)

        print('mean:', np.mean(times))
        print('std:', np.std(times))


if __name__ == '__main__':
    # Handle command-line arguments
    if len(sys.argv) < 2:
        print(f'USAGE: python3 {sys.argv[0]} <framework> [num_processes]')
        print('    framework: ray, lewicki, mp')
        print('    num_processes: space-separated list of processes for each run. Default: 1')
        print(f'EXAMPLE: python3 {sys.argv[0]} ray 2 4 8 16')
        sys.exit(0)

    try:
        num_processes = list(map(int, sys.argv[2:])) or [1]
    except ValueError:
        print('The value(s) for num_processes must be integers')
        sys.exit(0)

    framework = sys.argv[1]
    if framework in ['ray', 'lewicki', 'mp']:
        run_experiment(framework, num_processes)
    else:
        print('Unrecognized framework:', framework)
