#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
image=hw3-resnet18:23.12
server=hw3-triton
sdk=nvcr.io/nvidia/tritonserver:23.12-py3-sdk
command=${1:-start}
shift || true
prepare_images() {
  for reference in nvcr.io/nvidia/tritonserver:23.12-py3 "$sdk"; do
    if ! docker image inspect "$reference" >/dev/null 2>&1; then
      if ! docker pull "$reference"; then
        mirror="nvcr.m.daocloud.io/${reference#nvcr.io/}"
        echo "NVIDIA Registry недоступен. Скачиваем публичный образ через $mirror"
        docker pull "$mirror"
        docker tag "$mirror" "$reference"
      fi
    fi
  done
}
wait_ready() {
  for _ in {1..60}; do
    if curl -fsS http://localhost:8000/v2/models/resnet18/ready >/dev/null 2>&1; then
      echo 'ResNet-18 готов: HTTP :8000, gRPC :8001, метрики :8002'
      return
    fi
    sleep 1
  done
  docker logs "$server"
  echo 'Сервер не стал готов за 60 секунд' >&2
  exit 1
}
case "$command" in
  build)
    prepare_images
    docker build -t "$image" .
    ;;
  check-gpu)
    docker run --rm --gpus all ubuntu:22.04 nvidia-smi
    ;;
  start)
    prepare_images
    docker build -t "$image" .
    docker run --rm -d --name "$server" --gpus all --net host --shm-size=1g "$image"
    wait_ready
    ;;
  stop)
    docker stop "$server"
    ;;
  logs)
    docker logs "$server"
    ;;
  client)
    docker run --rm --net host -v "$PWD:/work" "$image" python3 client.py "$@"
    ;;
  check)
    docker run --rm --gpus all --net host -v "$PWD:/work" "$image" python3 check.py "$@"
    ;;
  perf)
    docker run --rm --net host -v "$PWD:/work" -w /work "$sdk" python3 perf.py "$@"
    ;;
  analyze)
    if [[ -n "$(docker ps --filter name="^/${server}$" -q)" ]]; then
      echo 'Останови сервер перед Model Analyzer: bash run.sh stop' >&2
      exit 1
    fi
    mkdir -p results/model_analyzer
    docker run --rm --gpus all --net host --shm-size=1g -v "$PWD:/work" "$image" \
      model-analyzer --mode offline profile -f configs/analyzer.yaml "$@"
    ;;
  optimized)
    if [[ ! -f configs/optimized.pbtxt ]]; then
      echo 'Сначала запусти analyze и select_config.py: нужен configs/optimized.pbtxt' >&2
      exit 1
    fi
    prepare_images
    docker build -t "$image" .
    docker run --rm -d --name "$server" --gpus all --net host --shm-size=1g \
      -v "$PWD/configs/optimized.pbtxt:/models/resnet18/config.pbtxt:ro" "$image"
    wait_ready
    ;;
  *)
    echo 'Команды: start, build, stop, logs, check-gpu, client, check, perf, analyze, optimized' >&2
    exit 1
    ;;
esac
