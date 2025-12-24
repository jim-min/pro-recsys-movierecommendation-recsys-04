#!/bin/bash

# BERT4Rec Training and Inference Script
# Usage: ./run_bert4rec.sh [train|predict|both]

set -e  # Exit on error

MODE=${1:-both}  # Default: both
CFG_FILE=${1:-bert4rec_v2}

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
        echo "Usage: ./run_bert4rec.sh [train|predict|both]"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Done!"
echo "========================================="
