#!/bin/bash

# MultiVAE Training and Inference Script
# Usage: ./run_multi_vae.sh [train|predict|both|clean] [config_file]

set -e  # Exit on error

MODE=${1:-both}  # Default: both
CFG_FILE=${2:-multi_vae_v2}
CACHE_DIR="${HOME}/.cache/recsys"

echo "========================================="
echo "MultiVAE Runner"
echo "========================================="
echo "Mode: $MODE"
echo "Config: $CFG_FILE"
echo ""

case $MODE in
    clean)
        echo "Cleaning cache..."
        if [ -d "$CACHE_DIR" ]; then
            echo "Removing cache directory: $CACHE_DIR"
            rm -rf "$CACHE_DIR"
            echo "✅ Cache cleaned successfully!"
        else
            echo "ℹ️  Cache directory not found: $CACHE_DIR"
        fi
        ;;

    train)
        echo "Starting training..."
        python train_multi_vae.py -cn $CFG_FILE
        ;;

    predict)
        echo "Starting inference..."
        if [ -f "predict_multi_vae.py" ]; then
            python predict_multi_vae.py -cn $CFG_FILE
        else
            echo "Warning: predict_multi_vae.py not found"
            echo "Skipping prediction step"
        fi
        ;;

    both)
        echo "Starting training..."
        python train_multi_vae.py -cn $CFG_FILE

        echo ""
        echo "Training completed. Starting inference..."
        if [ -f "predict_multi_vae.py" ]; then
            python predict_multi_vae.py -cn $CFG_FILE
        else
            echo "Warning: predict_multi_vae.py not found"
            echo "Skipping prediction step"
        fi
        ;;

    clean-train)
        echo "Cleaning cache and starting training..."
        if [ -d "$CACHE_DIR" ]; then
            echo "Removing cache directory: $CACHE_DIR"
            rm -rf "$CACHE_DIR"
            echo "✅ Cache cleaned!"
        fi
        echo ""
        echo "Starting training..."
        python train_multi_vae.py -cn $CFG_FILE
        ;;

    clean-both)
        echo "Cleaning cache and running full pipeline..."
        if [ -d "$CACHE_DIR" ]; then
            echo "Removing cache directory: $CACHE_DIR"
            rm -rf "$CACHE_DIR"
            echo "✅ Cache cleaned!"
        fi
        echo ""
        echo "Starting training..."
        python train_multi_vae.py -cn $CFG_FILE

        echo ""
        echo "Training completed. Starting inference..."
        if [ -f "predict_multi_vae.py" ]; then
            python predict_multi_vae.py -cn $CFG_FILE
        else
            echo "Warning: predict_multi_vae.py not found"
            echo "Skipping prediction step"
        fi
        ;;

    *)
        echo "Invalid mode: $MODE"
        echo ""
        echo "Usage: ./run_multi_vae.sh [mode] [config_file]"
        echo ""
        echo "Modes:"
        echo "  train         - Train only"
        echo "  predict       - Predict only"
        echo "  both          - Train + Predict (default)"
        echo "  clean         - Clean cache only"
        echo "  clean-train   - Clean cache + Train"
        echo "  clean-both    - Clean cache + Train + Predict"
        echo ""
        echo "Examples:"
        echo "  ./run_multi_vae.sh clean"
        echo "  ./run_multi_vae.sh clean-train multi_vae_v2"
        echo "  ./run_multi_vae.sh both multi_vae_v2"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Done!"
echo "========================================="
