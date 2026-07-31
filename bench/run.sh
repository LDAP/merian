#!/usr/bin/env bash
# Runs each named config and prints the "measured (Reduce)" GPU section stats.
# Configs are copied to a temp dir first: the runner sets the store path to the config
# it was given and would otherwise rewrite it.
set -u
BIN=build/merian-graph-run
FRAMES=${FRAMES:-1200}
WARMUP=${WARMUP:-400}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

printf '%-22s %9s %9s %9s %9s\n' config mean_ms p50_ms min_ms max_ms
for cfg in "$@"; do
    name=$(basename "$cfg" .json)
    cp "$cfg" "$TMP/$name.json"
    out=$(timeout 300 "$BIN" "$TMP/$name.json" --frames="$FRAMES" --warmup="$WARMUP" \
              --validation=off 2>&1)
    line=$(printf '%s\n' "$out" | grep "measured (Reduce)")
    if [ -z "$line" ]; then
        printf '%-22s %s\n' "$name" "FAILED"
        printf '%s\n' "$out" | grep -iE "error|failed" | head -3
        continue
    fi
    mean=$(printf '%s' "$line" | grep -oP 'mean=\s*\K[0-9.]+')
    p50=$(printf '%s' "$line" | grep -oP 'p50=\s*\K[0-9.]+')
    min=$(printf '%s' "$line" | grep -oP 'min=\s*\K[0-9.]+')
    max=$(printf '%s' "$line" | grep -oP 'max=\s*\K[0-9.]+')
    printf '%-22s %9s %9s %9s %9s\n' "$name" "$mean" "$p50" "$min" "$max"
done
