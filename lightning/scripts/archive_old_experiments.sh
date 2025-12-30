#!/bin/bash
# 예전 실험 로그를 별도 디렉토리로 이동하여 HParams 표시 문제 해결

CURRENT_DIR="./saved/tensorboard_logs/bert4rec"
ARCHIVE_DIR="./saved/tensorboard_logs/bert4rec_old"

echo "Moving old experiments to archive..."
echo "This will move experiments that don't have the new hparams structure."
echo ""

# Archive 디렉토리 생성
mkdir -p "$ARCHIVE_DIR"

# 처음 6개 실험 이동 (예전 hparams 구조)
OLD_EXPERIMENTS=(
    "2025-12-22/15-18-18"
    "2025-12-22/15-39-00"
    "2025-12-22/16-34-25"
    "2025-12-24/09-23-51"
    "2025-12-24/11-47-42"
    "2025-12-26/01-36-29"
)

for exp in "${OLD_EXPERIMENTS[@]}"; do
    if [ -d "$CURRENT_DIR/$exp" ]; then
        echo "Moving: $exp"
        # 날짜 디렉토리 생성
        date_dir=$(dirname "$exp")
        mkdir -p "$ARCHIVE_DIR/$date_dir"
        # 이동
        mv "$CURRENT_DIR/$exp" "$ARCHIVE_DIR/$exp"
    fi
done

# 빈 날짜 디렉토리 정리
find "$CURRENT_DIR" -type d -empty -delete

echo ""
echo "✓ Done! Old experiments archived to: $ARCHIVE_DIR"
echo ""
echo "Now TensorBoard will show all hparams for the remaining experiments."
echo "To view old experiments separately:"
echo "  tensorboard --logdir=$ARCHIVE_DIR"
