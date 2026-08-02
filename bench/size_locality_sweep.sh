#!/usr/bin/env bash
# Composition check: MC table size x Morton locality bits (LC fixed at preset size).
# Points are mc_slots:mc_bits:lc_bits triples.
set -u
BIN=build/merian-graph-run
QUAKE=${QUAKE:-subprojects/merian-plugin-quake/quake.json}
FRAMES=${FRAMES:-400}
WARMUP=${WARMUP:-128}
LOGDIR=${LOGDIR:-bench/size_locality_logs}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$LOGDIR"

if [ $# -eq 0 ]; then
    set -- 8000009:0:0 8000009:3:3 4000037:0:0 4000037:3:3 2000003:0:0 2000003:3:3
fi

cp "$QUAKE" "$TMP/quake.json"

printf '%-22s %9s %9s %9s %9s\n' "mc_slots:mc:lc" mean_ms p50_ms min_ms max_ms
for triple in "$@"; do
    IFS=: read -r slots mcb lcb <<< "$triple"
    python3 - "$TMP/patch.json" "$slots" "$mcb" "$lcb" <<'EOF'
import json, sys
json.dump({"nodes": {"render": {"properties": {
    "mc": {"adaptive grid buf size": int(sys.argv[2]), "locality bits": int(sys.argv[3])},
    "lc": {"LC locality bits": int(sys.argv[4])}}}}},
    open(sys.argv[1], "w"))
EOF
    out=$(timeout 300 "$BIN" "$TMP/quake.json" --renderer mcpg --merge "$TMP/patch.json" \
              --frames="$FRAMES" --warmup="$WARMUP" --validation=off \
              +timedemo demo1 2>&1)
    printf '%s\n' "$out" > "$LOGDIR/${slots}_${mcb}_${lcb}.log"
    line=$(printf '%s\n' "$out" | grep "nodes/render (" | head -1)
    if [ -z "$line" ]; then
        printf '%-22s %s\n' "$triple" "FAILED (see $LOGDIR/${slots}_${mcb}_${lcb}.log)"
        continue
    fi
    mean=$(printf '%s' "$line" | grep -oP 'mean=\s*\K[0-9.]+')
    p50=$(printf '%s' "$line" | grep -oP 'p50=\s*\K[0-9.]+')
    min=$(printf '%s' "$line" | grep -oP 'min=\s*\K[0-9.]+')
    max=$(printf '%s' "$line" | grep -oP 'max=\s*\K[0-9.]+')
    printf '%-22s %9s %9s %9s %9s\n' "$triple" "$mean" "$p50" "$min" "$max"
done
