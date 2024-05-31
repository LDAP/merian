## Utilities

Merian provides utilities for some daily duties of a Vulkan programmer, to name a few:


- `Camera`: Helper class to calculate view and projection matrices.
- `CameraAnimator`: Helper class to smooth camera motion.
- `CameraController`: Helper class to control a camera with high level commands.
- `Configuration`: An "immediate-mode" configuration API with implementation for ImGUI as well as JSON dumping and loading.
- `FileLoader`: Helper class to find and load files from search paths.
- `InputController`: An interface for keyboard and mouse inputs.
- `Profiler`: A profiler for CPU and GPU processing
- `ThreadPool`: A simple thread pool. Merian initializes a thread pool by default.

There are many more, have a look into `src/merian/utils` and `src/merian/vk/utils` 
