#!/bin/bash

# BERT4Rec Training and Inference Script
# Usage: ./run_bert4rec.sh [train|predict|both] [config_file]
# Examples:
#   ./run_bert4rec.sh train bert4rec_v2
#   ./run_bert4rec.sh predict
#   ./run_bert4rec.sh both

set -e  # Exit on error

MODE=${1:-both}  # Default: both
CFG_FILE=${2:-bert4rec_v2}  # Default: bert4rec_v2

echo "========================================="
echo "BERT4Rec Runner"
echo "========================================="
echo "Mode: $MODE"
echo "Config: $CFG_FILE"
echo ""

case $MODE in
    train)
        echo "Starting training..."
        python train_bert4rec.py -cn $CFG_FILE
        ;;

    predict)
        echo "Starting inference..."
        python predict_bert4rec.py -cn $CFG_FILE
        ;;

    both)
        echo "Starting training..."
        python train_bert4rec.py -cn $CFG_FILE

        echo ""
        echo "Training completed. Starting inference..."
        python predict_bert4rec.py -cn $CFG_FILE
        ;;

    *)
        echo "Invalid mode: $MODE"
        echo "Usage: ./run_bert4rec.sh [train|predict|both] [config_file]"
        echo "Examples:"
        echo "  ./run_bert4rec.sh train bert4rec_v2"
        echo "  ./run_bert4rec.sh predict"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Done!"
echo "========================================="
