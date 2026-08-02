#!/usr/bin/env bash
# Sweeps the MCPG/LC hash-grid table sizes on the quake mcpg timedemo and prints the
# render-pass timing per point. Pass sweep points as MC:LC slot-count pairs, e.g.
#   bench/grid_sweep.sh 32777259:8000009 1000003:1000003
# No args runs the default sweep. The config is copied to a temp dir first: the runner
# sets the store path to the config it was given and would otherwise rewrite it.
set -u
BIN=build/merian-graph-run
QUAKE=${QUAKE:-subprojects/merian-plugin-quake/quake.json}
FRAMES=${FRAMES:-400}
WARMUP=${WARMUP:-128}
SECTION=${SECTION:-nodes/render }
LOGDIR=${LOGDIR:-bench/grid_sweep_logs}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$LOGDIR"

# default sweep: MC at fixed LC, then LC at small MC, then combined IC-fit candidates
if [ $# -eq 0 ]; then
    set -- 32777259:8000009 16000057:8000009 8000009:8000009 4000037:8000009 \
           2000003:8000009 1000003:8000009 500009:8000009 \
           1000003:4000037 1000003:2000003 1000003:1000003 1000003:500009 \
           1500007:500009 2000003:1000003
fi

cp "$QUAKE" "$TMP/quake.json"

printf '%-22s %9s %9s %9s %9s %9s\n' "mc:lc(slots)" total_mb mean_ms p50_ms min_ms max_ms
for pair in "$@"; do
    mc=${pair%%:*}
    lc=${pair##*:}
    total_mb=$(( (mc * 40 + lc * 16) / 1000000 ))
    python3 - "$TMP/patch.json" "$mc" "$lc" <<'EOF'
import json, sys
json.dump({"nodes": {"render": {"properties": {
    "mc": {"adaptive grid buf size": int(sys.argv[2])},
    "lc": {"LC buffer size": int(sys.argv[3])}}}}},
    open(sys.argv[1], "w"))
EOF
    out=$(timeout 300 "$BIN" "$TMP/quake.json" --renderer mcpg --merge "$TMP/patch.json" \
              --frames="$FRAMES" --warmup="$WARMUP" --validation=off \
              +timedemo demo1 2>&1)
    printf '%s\n' "$out" > "$LOGDIR/${mc}_${lc}.log"
    line=$(printf '%s\n' "$out" | grep -i "$SECTION" | grep "mean=" | head -1)
    if [ -z "$line" ]; then
        printf '%-22s %9s %s\n' "$pair" "$total_mb" "FAILED (see $LOGDIR/${mc}_${lc}.log)"
        continue
    fi
    mean=$(printf '%s' "$line" | grep -oP 'mean=\s*\K[0-9.]+')
    p50=$(printf '%s' "$line" | grep -oP 'p50=\s*\K[0-9.]+')
    min=$(printf '%s' "$line" | grep -oP 'min=\s*\K[0-9.]+')
    max=$(printf '%s' "$line" | grep -oP 'max=\s*\K[0-9.]+')
    printf '%-22s %9s %9s %9s %9s %9s\n' "$pair" "$total_mb" "$mean" "$p50" "$min" "$max"
done
