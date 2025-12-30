#!/bin/bash
# HParams를 제대로 표시하도록 최적화된 TensorBoard 시작 스크립트

LOGDIR="./saved/tensorboard_logs"
PORT=6006

echo "Starting TensorBoard with optimized settings for HParams..."
echo "Log directory: $LOGDIR"
echo "Port: $PORT"
echo ""

# TensorBoard 시작 옵션 설명:
# --reload_multifile=true : 여러 파일의 데이터를 다시 로드
# --reload_interval=30 : 30초마다 새로운 데이터 확인
# --samples_per_plugin : 각 플러그인에서 로드할 샘플 수 증가
# --max_reload_threads=4 : 파일 로드 스레드 수 증가
# --purge_orphaned_data=false : 고아 데이터를 삭제하지 않음 (HParams 유지)

tensorboard --logdir=$LOGDIR \
    --port=$PORT \
    --reload_multifile=true \
    --reload_interval=30 \
    --samples_per_plugin=hparams=10000 \
    --max_reload_threads=4 \
    --purge_orphaned_data=false \
    --bind_all
