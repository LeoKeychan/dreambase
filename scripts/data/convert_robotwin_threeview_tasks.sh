#!/bin/bash
# Convert RoboTwin clean_50 demos into DreamBase three-view LeRobot datasets.

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        *)
            echo "ERROR: unknown argument: $1"
            echo "Supported argument: --dry-run"
            exit 1
            ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMZERO_ROOT="${DREAMZERO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
if [ ! -d "$DREAMZERO_ROOT/groot" ]; then
    echo "ERROR: No groot/ under $DREAMZERO_ROOT. Set DREAMZERO_ROOT to the dreambase repo root."
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-/meta_eon_cfs/home/szj/miniconda3/envs/dream/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python env not found: $PYTHON_BIN"
    exit 1
fi

ROBOTWIN_RAW_ROOT="${ROBOTWIN_RAW_ROOT:-/meta_eon_cfs/home/lqc/dataset/robotwin/clean_50}"  # 要改成自己的 robotwin 的 clean 路径
ROBOTWIN_OUTPUT_ROOT="${ROBOTWIN_OUTPUT_ROOT:-/meta_eon_cfs/home/lqc/dataset/robotwin/dreamzero_robotwin_baseline}"  # 要改成自己的输出路径
ROBOTWIN_DEMO_DIR="${ROBOTWIN_DEMO_DIR:-aloha-agilex_clean_50}"  # 确认下载的数据集里的 clean 是不是这个名字
EPISODES="${EPISODES:-0-49}"
REQUIRE_EPISODES="${REQUIRE_EPISODES:-50}"
OVERWRITE="${OVERWRITE:-0}"
SOURCE_WIDTH="${SOURCE_WIDTH:-320}"
SOURCE_HEIGHT="${SOURCE_HEIGHT:-176}"
FPS="${FPS:-30}"

if [ "${ROBOTWIN_TASKS:-all}" = "all" ]; then
    ROBOTWIN_TASK_ARRAY=()
    while IFS= read -r demo_dir; do
        ROBOTWIN_TASK_ARRAY+=("$(basename "$(dirname "$demo_dir")")")
    done < <(find "$ROBOTWIN_RAW_ROOT" -mindepth 2 -maxdepth 2 -type d -name "$ROBOTWIN_DEMO_DIR" | sort)
elif [ -n "${ROBOTWIN_TASKS:-}" ]; then
    ROBOTWIN_TASKS_NORMALIZED="${ROBOTWIN_TASKS//,/ }"
    read -r -a ROBOTWIN_TASK_ARRAY <<< "$ROBOTWIN_TASKS_NORMALIZED"
else
    echo "ERROR: no RoboTwin tasks selected."
    exit 1
fi

if [ "${#ROBOTWIN_TASK_ARRAY[@]}" -eq 0 ]; then
    echo "ERROR: no RoboTwin tasks selected."
    exit 1
fi

cd "$DREAMZERO_ROOT"
export PYTHONPATH="$DREAMZERO_ROOT"

overwrite_args=()
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ]; then
    overwrite_args=(--overwrite)
fi

echo "RoboTwin raw root: $ROBOTWIN_RAW_ROOT"
echo "Output root: $ROBOTWIN_OUTPUT_ROOT"
echo "Demo dir: $ROBOTWIN_DEMO_DIR"
echo "Episodes: $EPISODES"
echo "Source size: ${SOURCE_WIDTH}x${SOURCE_HEIGHT}"
echo "Dry run: $DRY_RUN"
echo "Tasks (${#ROBOTWIN_TASK_ARRAY[@]}): ${ROBOTWIN_TASK_ARRAY[*]}"

for task in "${ROBOTWIN_TASK_ARRAY[@]}"; do
    demo_root="$ROBOTWIN_RAW_ROOT/$task/$ROBOTWIN_DEMO_DIR"
    output_root="$ROBOTWIN_OUTPUT_ROOT/$task"

    if [ ! -d "$demo_root/data" ]; then
        echo "ERROR: missing demo data for task '$task': $demo_root/data"
        exit 1
    fi

    demo_count="$(find "$demo_root/data" -maxdepth 1 -name 'episode*.hdf5' | wc -l)"
    if [ "$REQUIRE_EPISODES" -gt 0 ] && [ "$demo_count" -lt "$REQUIRE_EPISODES" ]; then
        echo "ERROR: task '$task' has only $demo_count demo episodes, expected at least $REQUIRE_EPISODES"
        exit 1
    fi

    if [ "$DRY_RUN" = "1" ] || [ "$DRY_RUN" = "true" ]; then
        echo "DRY-RUN: would convert task '$task' -> $output_root"
        continue
    fi

    echo "Converting task '$task' -> $output_root"
    "$PYTHON_BIN" scripts/data/convert_robotwin_threeview.py \
        --demo-root "$demo_root" \
        --output-root "$output_root" \
        --episodes "$EPISODES" \
        --fps "$FPS" \
        --source-width "$SOURCE_WIDTH" \
        --source-height "$SOURCE_HEIGHT" \
        "${overwrite_args[@]}"
done

echo "Done converting ${#ROBOTWIN_TASK_ARRAY[@]} DreamBase three-view RoboTwin tasks."
echo "Train with:"
echo "  ROBOTWIN_DATASET_ROOT=$ROBOTWIN_OUTPUT_ROOT bash scripts/train/robotwin_threeview_baseline_training.sh"
