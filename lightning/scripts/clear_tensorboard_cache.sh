#!/bin/bash
# TensorBoard 캐시를 삭제하고 재시작하는 스크립트

echo "Clearing TensorBoard cache..."

# TensorBoard 캐시 디렉토리 삭제
rm -rf /tmp/.tensorboard-info/
rm -rf ~/.tensorboard/

# 혹시 실행 중인 TensorBoard 프로세스 종료
pkill -f tensorboard

echo "Cache cleared!"
echo ""
echo "Now you can start TensorBoard with:"
echo "tensorboard --logdir=./saved/tensorboard_logs --reload_multifile=true --reload_interval=30"
