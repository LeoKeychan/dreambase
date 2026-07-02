#!/bin/bash
# DreamZero RoboTwin three-view baseline fine-tuning.
#
# This uses the converted RoboTwin dataset's head/left/right cameras only:
#   [input_head | input_right]
#   [input_left | black]
# and trains the original DreamZero video/action objective to predict the same
# three-view mosaic video, not the single target_front cross-view video.
#
# Default data layout for 16-train:
#   /meta_eon_cfs/home/lqc/dataset/robotwin_depth/<task>/dreamzero_robotwin_crossview

set -euo pipefail

export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_dreamzero}"

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

ROBOTWIN_DEPTH_ROOT="${ROBOTWIN_DEPTH_ROOT:-/meta_eon_cfs/home/lqc/dataset/robotwin_depth}"
ROBOTWIN_OUTPUT_NAME="${ROBOTWIN_OUTPUT_NAME:-dreamzero_robotwin_crossview}"
OUTPUT_DIR="${OUTPUT_DIR:-$DREAMZERO_ROOT/checkpoints/robotwin_test_speed}"
NUM_GPUS="${NUM_GPUS:-8}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((NUM_GPUS * PER_DEVICE_BS))}"
MAX_STEPS="${MAX_STEPS:-80000}"
SAVE_STEPS="${SAVE_STEPS:-30}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-zero2}"

# Bucket batching lets PER_DEVICE_BS>1 work with RoboTwin samples that have
# different natural chunk counts. Set ROBOTWIN_BUCKET_BATCHING=false to restore
# the old DataLoader batch path.
ROBOTWIN_BUCKET_BATCHING="${ROBOTWIN_BUCKET_BATCHING:-false}"
ROBOTWIN_BUCKET_MIN_CHUNKS="${ROBOTWIN_BUCKET_MIN_CHUNKS:-2}"

SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-false}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"

WAN21_CKPT_DIR="${WAN21_CKPT_DIR:-$DREAMZERO_ROOT/checkpoints/Wan2.1-I2V-14B-480P}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$DREAMZERO_ROOT/checkpoints/umt5-xxl}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-$DREAMZERO_ROOT/checkpoints/DreamZero-AgiBot}"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/train_robotwin_threeview_baseline.log}"

DEFAULT_ROBOTWIN_TASKS=(
    beat_block_hammer
    click_alarmclock
    dump_bin_bigbin
    grab_roller
    handover_block
    hanging_mug
    lift_pot
    move_can_pot
    move_playingcard_away
    open_laptop
    open_microwave
    pick_diverse_bottles
    pick_dual_bottles
    shake_bottle
    place_can_basket
    place_object_stand
)

if [ -n "${ROBOTWIN_TASKS:-}" ]; then
    ROBOTWIN_TASKS_NORMALIZED="${ROBOTWIN_TASKS//,/ }"
    read -r -a ROBOTWIN_TASK_ARRAY <<< "$ROBOTWIN_TASKS_NORMALIZED"
else
    ROBOTWIN_TASK_ARRAY=("${DEFAULT_ROBOTWIN_TASKS[@]}")
fi

if [ -n "${ROBOTWIN_DATA_ROOTS:-}" ]; then
    HYDRA_ROBOTWIN_DATA_ROOTS="$ROBOTWIN_DATA_ROOTS"
elif [ -n "${ROBOTWIN_DATA_ROOT:-}" ]; then
    if [ ! -d "$ROBOTWIN_DATA_ROOT" ]; then
        echo "ERROR: RoboTwin dataset not found at $ROBOTWIN_DATA_ROOT"
        exit 1
    fi
    if [ ! -f "$ROBOTWIN_DATA_ROOT/meta/modality.json" ]; then
        echo "ERROR: missing $ROBOTWIN_DATA_ROOT/meta/modality.json"
        exit 1
    fi
    HYDRA_ROBOTWIN_DATA_ROOTS="[$ROBOTWIN_DATA_ROOT]"
    ROBOTWIN_TASK_ARRAY=("single_dataset")
else
    ROBOTWIN_DATA_ROOT_ARRAY=()
    for task in "${ROBOTWIN_TASK_ARRAY[@]}"; do
        task_root="$ROBOTWIN_DEPTH_ROOT/$task/$ROBOTWIN_OUTPUT_NAME"
        if [ ! -d "$task_root" ]; then
            echo "ERROR: DreamBase RoboTwin dataset not found for task '$task': $task_root"
            echo "Expected converted RoboTwin dataset: dreamzero_robotwin_crossview"
            exit 1
        fi
        if [ ! -f "$task_root/meta/modality.json" ]; then
            echo "ERROR: missing $task_root/meta/modality.json"
            exit 1
        fi
        ROBOTWIN_DATA_ROOT_ARRAY+=("$task_root")
    done

    HYDRA_ROBOTWIN_DATA_ROOTS="["
    for root in "${ROBOTWIN_DATA_ROOT_ARRAY[@]}"; do
        if [ "$HYDRA_ROBOTWIN_DATA_ROOTS" != "[" ]; then
            HYDRA_ROBOTWIN_DATA_ROOTS+=","
        fi
        HYDRA_ROBOTWIN_DATA_ROOTS+="$root"
    done
    HYDRA_ROBOTWIN_DATA_ROOTS+="]"
fi

for numeric_var in NUM_GPUS PER_DEVICE_BS GLOBAL_BATCH_SIZE ROBOTWIN_BUCKET_MIN_CHUNKS; do
    numeric_value="${!numeric_var}"
    if ! [[ "$numeric_value" =~ ^[0-9]+$ ]] || [ "$numeric_value" -lt 1 ]; then
        echo "ERROR: $numeric_var must be a positive integer, got '$numeric_value'"
        exit 1
    fi
done

PER_STEP_BATCH_SIZE=$((NUM_GPUS * PER_DEVICE_BS))
if [ $((GLOBAL_BATCH_SIZE % PER_STEP_BATCH_SIZE)) -ne 0 ]; then
    echo "ERROR: GLOBAL_BATCH_SIZE must be divisible by NUM_GPUS * PER_DEVICE_BS"
    echo "  GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
    echo "  NUM_GPUS * PER_DEVICE_BS=$PER_STEP_BATCH_SIZE"
    exit 1
fi
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / PER_STEP_BATCH_SIZE))

if [ ! -d "$WAN21_CKPT_DIR" ]; then
    echo "ERROR: Wan2.1 checkpoint not found at $WAN21_CKPT_DIR"
    exit 1
fi
if [ ! -d "$TOKENIZER_DIR" ]; then
    echo "ERROR: tokenizer not found at $TOKENIZER_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$DREAMZERO_ROOT"

unset PYTHONPATH
export PYTHONPATH="$DREAMZERO_ROOT"

GROOT_IMPORT_PATH="$("$PYTHON_BIN" -c 'import pathlib, groot; print(pathlib.Path(groot.__file__).resolve())')"
case "$GROOT_IMPORT_PATH" in
    "$DREAMZERO_ROOT"/groot/*)
        echo "Using groot from: $GROOT_IMPORT_PATH"
        ;;
    *)
        echo "ERROR: imported groot from unexpected path: $GROOT_IMPORT_PATH"
        echo "Expected it under: $DREAMZERO_ROOT/groot"
        exit 1
        ;;
esac

echo "RoboTwin depth root: $ROBOTWIN_DEPTH_ROOT"
echo "RoboTwin train tasks (${#ROBOTWIN_TASK_ARRAY[@]}): ${ROBOTWIN_TASK_ARRAY[*]}"
echo "RoboTwin data roots: $HYDRA_ROBOTWIN_DATA_ROOTS"
echo "Per-device batch size: $PER_DEVICE_BS"
echo "Global batch size: $GLOBAL_BATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACCUM_STEPS"
echo "RoboTwin bucket batching: $ROBOTWIN_BUCKET_BATCHING"
echo "RoboTwin bucket min chunks: $ROBOTWIN_BUCKET_MIN_CHUNKS"
echo "Learning rate: $LEARNING_RATE"
echo "Save only model: $SAVE_ONLY_MODEL"
echo "Output dir: $OUTPUT_DIR"

# train api key
export WANDB_API_KEY=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
wandb login

"$PYTHON_BIN" -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone \
    groot/vla/experiment/experiment.py \
    report_to=none \
    data=dreamzero/robotwin_crossview_relative \
    wandb_project=dreambase_robotwin \
    train_architecture=full \
    num_frames=33 \
    action_horizon=24 \
    num_views=4 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate="$LEARNING_RATE" \
    training_args.deepspeed="groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json" \
    save_steps="$SAVE_STEPS" \
    training_args.warmup_ratio=0.02 \
    output_dir="$OUTPUT_DIR" \
    per_device_train_batch_size="$PER_DEVICE_BS" \
    global_batch_size="$GLOBAL_BATCH_SIZE" \
    max_steps="$MAX_STEPS" \
    weight_decay=1e-5 \
    save_total_limit="$SAVE_TOTAL_LIMIT" \
    save_only_model="$SAVE_ONLY_MODEL" \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    robotwin_bucket_batching="$ROBOTWIN_BUCKET_BATCHING" \
    robotwin_bucket_min_chunks="$ROBOTWIN_BUCKET_MIN_CHUNKS" \
    robotwin_image_resolution_width=320 \
    robotwin_image_resolution_height=176 \
    max_state_dim=64 \
    max_action_dim=32 \
    save_lora_only=false \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    robotwin_depth_root="$ROBOTWIN_DEPTH_ROOT" \
    robotwin_output_name="$ROBOTWIN_OUTPUT_NAME" \
    robotwin_data_roots="$HYDRA_ROBOTWIN_DATA_ROOTS" \
    dit_version="$WAN21_CKPT_DIR" \
    text_encoder_pretrained_path="$WAN21_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="$WAN21_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="$WAN21_CKPT_DIR/Wan2.1_VAE.pth" \
    tokenizer_path="$TOKENIZER_DIR" \
    pretrained_model_path="$PRETRAINED_MODEL_PATH" \
    ++action_head_cfg.config.target_video_width=640 \
    ++action_head_cfg.config.target_video_height=352 \
    ++action_head_cfg.config.skip_component_loading=true \
    2>&1 | tee "$LOG_FILE"
