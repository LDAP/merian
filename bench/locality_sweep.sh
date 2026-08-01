#!/usr/bin/env bash
# A/Bs hash-grid Morton locality bits (mc:lc pairs) at full table sizes on the quake mcpg
# scenario. SCENE=demo (timedemo demo1) or static (+map, motionless camera).
set -u
BIN=build/merian-graph-run
QUAKE=${QUAKE:-subprojects/merian-plugin-quake/quake.json}
FRAMES=${FRAMES:-400}
WARMUP=${WARMUP:-128}
SCENE=${SCENE:-demo}
LOGDIR=${LOGDIR:-bench/locality_sweep_logs}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$LOGDIR"

if [ "$SCENE" = static ]; then
    SCENE_ARGS=(+map e1m3)
else
    SCENE_ARGS=(+timedemo demo1)
fi

if [ $# -eq 0 ]; then
    set -- 0:0 1:0 2:0 3:0 4:0 0:3 2:2 3:3 4:4
fi

cp "$QUAKE" "$TMP/quake.json"

printf '%-12s %9s %9s %9s %9s\n' "mc:lc(bits)" mean_ms p50_ms min_ms max_ms
for pair in "$@"; do
    mc=${pair%%:*}
    lc=${pair##*:}
    python3 - "$TMP/patch.json" "$mc" "$lc" <<'EOF'
import json, sys
json.dump({"nodes": {"render": {"properties": {
    "mc": {"locality bits": int(sys.argv[2])},
    "lc": {"LC locality bits": int(sys.argv[3])}}}}},
    open(sys.argv[1], "w"))
EOF
    out=$(timeout 300 "$BIN" "$TMP/quake.json" --renderer mcpg --merge "$TMP/patch.json" \
              --frames="$FRAMES" --warmup="$WARMUP" --validation=off \
              "${SCENE_ARGS[@]}" 2>&1)
    printf '%s\n' "$out" > "$LOGDIR/${SCENE}_${mc}_${lc}.log"
    line=$(printf '%s\n' "$out" | grep "nodes/render (" | head -1)
    if [ -z "$line" ]; then
        printf '%-12s %s\n' "$pair" "FAILED (see $LOGDIR/${SCENE}_${mc}_${lc}.log)"
        continue
    fi
    mean=$(printf '%s' "$line" | grep -oP 'mean=\s*\K[0-9.]+')
    p50=$(printf '%s' "$line" | grep -oP 'p50=\s*\K[0-9.]+')
    min=$(printf '%s' "$line" | grep -oP 'min=\s*\K[0-9.]+')
    max=$(printf '%s' "$line" | grep -oP 'max=\s*\K[0-9.]+')
    printf '%-12s %9s %9s %9s %9s\n' "$pair" "$mean" "$p50" "$min" "$max"
done
