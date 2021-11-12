#!/usr/bin/python3

import colorsys
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
from PIL import Image

# Experiment parameters
NUM_TRIALS = 3

# Animation parameters
FPS = 15
NUM_SEC = 20
ZOOM_RATE = 0.99
MAX_ITER = 1000

# Image parameters
XY_PROP = 3 / 2
Y_DIM = 512
X_DIM = int(XY_PROP * Y_DIM)
IMG_DIM = (X_DIM, Y_DIM)

# Complex plane parameters
CENTER = (-0.7845, -0.1272)
Y_LENGTH = 2
X_LENGTH = XY_PROP * Y_LENGTH

### Fractal utility functions ###

def rgb(i):
    """Convert 8-bit integer to unique color in gradient"""
    color = 255 * np.array(colorsys.hsv_to_rgb(i / 255.0, 1.0, 0.5))
    return tuple(color.astype(int))

def get_windows(scale=1):
    """Compute the complex xy-viewport at a particular zoom level"""
    x_window = [CENTER[0] - scale * X_LENGTH / 2,
                CENTER[0] + scale * X_LENGTH / 2]
    y_window = [CENTER[1] - scale * Y_LENGTH / 2,
                CENTER[1] + scale * Y_LENGTH / 2]
    return x_window, y_window

def get_frame(frame_num):
    """Compute mandelbrot set membership for all pixels in view"""
    # Get matrix of pixels in view
    x_window, y_window = get_windows(scale=ZOOM_RATE ** frame_num)
    x = np.linspace(x_window[0], x_window[1], IMG_DIM[0]).reshape(1, IMG_DIM[0])
    y = np.linspace(y_window[0], y_window[1], IMG_DIM[1]).reshape(IMG_DIM[1], 1)

    # Initialize all pixels
    c = x + 1j * y
    z = np.zeros(c.shape, dtype=np.complex128)

    # Keep track of how many iterations a pixel is "active"
    # Once a pixel has been determined to not be in the set, it is no longer active
    iterations = np.zeros(z.shape, dtype=np.uint8)
    active = np.full(c.shape, True, dtype=bool)

    for i in range(MAX_ITER):
        z[active] = z[active]**2 + c[active]

        # A pixel cannot be part of the set if it has abs > sqrt 2
        diverged = np.greater(np.abs(z), 2.236, out=np.full(c.shape, False), where=active)
        iterations[diverged] = i
        active[diverged] = False

    iterations[active] = MAX_ITER + 1
    return frame_num, iterations

def save_results(results):
    # Create directory for storing results
    os.makedirs('frames', exist_ok=True)

    # Plot the first frame to get cmap
    im = plt.imshow(results[0][1], interpolation='none')

    for i, frame in results:
        c_frame = 255 * im.cmap(im.norm(frame))
        c_frame = np.delete(c_frame, -1, axis=2).astype(np.uint8)
        image = Image.fromarray(c_frame)
        image.save(f'frames/frame_{i:05}.png')

### Experiment function calls ###

def run_experiment(framework, num_processes):
    for N in num_processes:
        print(f'N = {N}')
        for trial in range(NUM_TRIALS):
            if framework == 'serial':
                # Run serial calculations
                start_time = time.time()
                results = run_serial()
                end_time = time.time()
            else:
                # Import correct pool implementation
                if framework == 'ray':
                    import ray
                    from ray.util.multiprocessing import Pool
                    ray.shutdown()
                    ray.init()
                    pool = Pool(N)
                elif framework == 'lewicki':
                    from lewicki import ActorPool
                    pool = ActorPool(N)
                elif framework == 'mp':
                    from multiprocessing import Pool
                    pool = Pool(N)

                # Run parallel calculations using pool
                start_time = time.time()
                results = run_parallel(pool)
                end_time = time.time()

            print(end_time - start_time)
    save_results(results)

def run_serial():
    return [get_frame(i) for i in range(FPS * NUM_SEC)]

def run_parallel(pool):
    return list(pool.map(get_frame, list(range(FPS * NUM_SEC))))


if __name__ == '__main__':
    # Handle command-line arguments
    if len(sys.argv) < 2:
        print(f'USAGE: python3 {sys.argv[0]} <framework> [num_processes]')
        print('    framework: ray, lewicki, serial')
        print('    num_processes: space-separated list of compute processes for each run. Default: 1')
        print(f'EXAMPLE: python3 {sys.argv[0]} ray 2 4 8 16')
        sys.exit(0)

    try:
        num_processes = list(map(int, sys.argv[2:])) or [1]
    except ValueError:
        print('The value(s) for num_processes must be integers')
        sys.exit(0)

    framework = sys.argv[1]
    if framework in ['ray', 'lewicki', 'mp', 'serial']:
        run_experiment(framework, num_processes)
    else:
        print('Unrecognized framework:', framework)
