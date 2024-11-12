### Performance profiling

Some merian functions take a `merian::ProfilerHandle`. However, you must enable profiling at compile-time with a meson option:

```bash
meson configure build -Dmerian:performance_profiling=false
```
