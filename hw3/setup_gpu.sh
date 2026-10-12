#!/usr/bin/env bash
# Выполнить один раз в Ubuntu/WSL: sudo bash setup_gpu.sh
set -euo pipefail
if [[ $EUID -ne 0 ]]; then
  echo 'Нужны права root: sudo bash setup_gpu.sh' >&2
  exit 1
fi
if [[ -n "$(docker ps -q)" ]]; then
  echo 'Сначала останови контейнеры: настройка требует перезапуска Docker.' >&2
  exit 1
fi
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey -o /tmp/hw3-nvidia-key
gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg /tmp/hw3-nvidia-key
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -qq
apt-get install -y --no-install-recommends nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
echo 'Toolkit установлен. Проверка GPU: bash run.sh check-gpu'
