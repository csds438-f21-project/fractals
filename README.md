
# Mandelbrot Fractal Zoom

This experiment applies the ActorPool from our Actor Model framework to generate frames for a video zooming in on the Mandelbrot set. The main Python script is `generate_fractal.py`.

```
USAGE: python3 generate_fractal.py <framework> [num_processes]
    framework: ray, lewicki, mp, serial
    num_processes: space-separated list of processes for each run. Default: 1
EXAMPLE: python3 generate_fractal.py ray 2 4 8 16
```