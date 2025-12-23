#!/bin/bash

# BERT4Rec Training and Inference Script
# Usage: ./run_bert4rec.sh [train|predict|both]

set -e  # Exit on error

MODE=${1:-both}  # Default: both

echo "========================================="
echo "BERT4Rec Runner"
echo "========================================="
echo "Mode: $MODE"
echo ""

case $MODE in
    train)
        echo "Starting training..."
        python train_bert4rec.py -cn bert4rec_improved
        ;;

    predict)
        echo "Starting inference..."
        python predict_bert4rec.py -cn bert4rec_improved
        ;;

    both)
        echo "Starting training..."
        python train_bert4rec.py -cn bert4rec_improved

        echo ""
        echo "Training completed. Starting inference..."
        python predict_bert4rec.py -cn bert4rec_improved
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
