#!/usr/bin/env bash
# Full transposition measurement sweep across grid sizes and terrain types.
# Intended to run overnight.
#
# Matrix:
#   Grid sizes: 5x5, 7x7, 9x9, 11x11
#   Terrain:    open (no walls/mud), walls (0.7), mud (0.1), walls+mud (0.7/0.1)
#   Sim counts: 50 200 554 1000 2000 (default sweep)
#   Games:      20 per sim count
#
# Output structure:
#   experiments/transposition_measurement/{size}_{terrain}/sims_{N}.json

set -euo pipefail

SCRIPT="scripts/measure_transpositions.py"
BASE_DIR="experiments/transposition_measurement"
GAMES=20
MAX_TURNS=50

# Grid size -> cheese count (scale with grid)
declare -A CHEESE
CHEESE[5]=5
CHEESE[7]=10
CHEESE[9]=16
CHEESE[11]=24

SIZES=(5 7 9 11)

# Terrain configs: name wall_density mud_density
TERRAINS=(
    "open 0.0 0.0"
    "walls 0.7 0.0"
    "mud 0.0 0.1"
    "walls_mud 0.7 0.1"
)

total=$((${#SIZES[@]} * ${#TERRAINS[@]}))
current=0

echo "=== Transposition Measurement Full Sweep ==="
echo "Grid sizes: ${SIZES[*]}"
echo "Terrains: open, walls, mud, walls+mud"
echo "Games per sim count: $GAMES"
echo "Total configurations: $total"
echo "Started: $(date)"
echo ""

for size in "${SIZES[@]}"; do
    cheese=${CHEESE[$size]}
    for terrain_spec in "${TERRAINS[@]}"; do
        read -r terrain_name wall_d mud_d <<< "$terrain_spec"
        current=$((current + 1))

        output_dir="${BASE_DIR}/${size}x${size}_${terrain_name}"
        echo "[$current/$total] ${size}x${size} ${terrain_name} (cheese=$cheese, walls=$wall_d, mud=$mud_d)"
        echo "  Output: $output_dir"
        echo "  Started: $(date)"

        uv run python "$SCRIPT" \
            --width "$size" --height "$size" \
            --cheese "$cheese" \
            --max-turns "$MAX_TURNS" \
            --games "$GAMES" \
            --wall-density "$wall_d" \
            --mud-density "$mud_d" \
            --output-dir "$output_dir"

        echo "  Finished: $(date)"
        echo ""
    done
done

echo "=== Sweep complete: $(date) ==="
