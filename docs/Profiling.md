### Performance profiling

Some merian functions take a `merian::ProfilerHandle`. Profiling is compiled in unless the meson option `performance_profiling` (default `true`) is switched off, `-Dmerian:performance_profiling=false` where merian is a subproject:

```bash
meson configure build -Dperformance_profiling=false
```
