#!/bin/bash
# Run training + evaluation for LeRF scenes using pre-processed data
# Usage: bash run_lerf.sh <gpu_id> <scene1> [scene2] ...

set -e
PYTHON=/opt/conda/envs/segment_then_splat/bin/python
PROJECT_ROOT=/root/data1/jinhyeok/Segment-then-Splat
DATA_ROOT=$PROJECT_ROOT/dataset/lerf-ovs
LABEL_ROOT=$DATA_ROOT/label

GPU_ID=$1
shift

for SCENE in "$@"; do
    echo "============================================================"
    echo "Processing lerf/$SCENE on GPU $GPU_ID"
    echo "============================================================"

    SCENE_DIR=$DATA_ROOT/$SCENE
    OUTPUT_DIR=$PROJECT_ROOT/output/$SCENE
    LABEL_DIR=$LABEL_ROOT/$SCENE/gt

    # Step 5: Training (pre-processed data already has obj_ids in PLY)
    echo "  Step 5: Training (40k iterations)..."
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON $PROJECT_ROOT/train.py \
        -s $SCENE_DIR/ -m $OUTPUT_DIR \
        --eval --iterations 40000 --num_sample_objects 3 \
        --densify_until_iter 20000 --partial_mask_iou 0.3

    # Step 6: Render
    echo "  Step 6: Rendering..."
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON $PROJECT_ROOT/render_objs.py \
        -m $OUTPUT_DIR/ --mode render --skip_train

    # Step 7: Evaluation
    echo "  Step 7: Evaluation..."
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON $PROJECT_ROOT/helpers/evaluation.py \
        --scene $SCENE_DIR/ --render_dir $OUTPUT_DIR/ \
        --label_dir $LABEL_DIR \
        --label_format lerf \
        --output_json $OUTPUT_DIR/eval_results.json

    echo "  Done: $SCENE"
done
