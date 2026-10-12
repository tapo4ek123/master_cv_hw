# Домашние задания по компьютерному зрению

В этом репозитории я собрал три работы: удаление фона с видео, обучение диффузионной модели на изображениях котов и запуск классификатора через Triton. У каждого задания есть свой код, инструкция, отчёт и результаты.

| Задание | Что сделано | Основной результат |
|---|---|---|
| [HW1 — удаление фона](hw1/README.md) | MediaPipe, веб-камера и видеофайл, три режима фона, работа на CPU | 50,43 FPS обработки на 640×480; 26,54 FPS полного цикла с камерой |
| [HW2 — диффузионная модель](hw2/README.md) | Обучение UNet, EMA, DDPM/DDIM, сравнение датасетов и размеров модели | Коты узнаваемы; DDIM 50 генерирует 8 изображений за 8,25 с против 159,35 с у DDPM 1000 |
| [HW3 — Triton](hw3/README.md) | ResNet-18, HTTP/gRPC, COUNTER/GAUGE, Performance Analyzer и Model Analyzer | Dynamic batching: 471,09 изображения/с и p99 35,32 мс при concurrency 16 |

## Структура

```text
hw1/  приложение удаления фона, модель, демо и замеры
hw2/  обучение и генерация, готовые веса, изображения и демо
hw3/  Dockerfile, модель Triton, клиенты и результаты профилирования
```

Зависимости зафиксированы отдельно для каждого задания. Команды ниже выполняются из корня репозитория. Подробные параметры и дополнительные режимы описаны в README соответствующего задания.

## Быстрый запуск HW1

Нужен Python 3.11 или 3.12. В Windows открыть PowerShell и выполнить:

```powershell
cd hw1
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1 --source 0 --compare
```

Скрипт создаст окружение и установит зависимости. Выход из приложения — Q или Esc. В [инструкции HW1](hw1/README.md) также есть установка вручную и запуск в Linux.

## Быстрый запуск HW2

Нужен Python 3.10–3.12. Для обучения и быстрого сэмплинга использовал NVIDIA GPU. В Linux/WSL:

```bash
cd hw2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python prepare_weights.py
python sample.py --compare --demo --demo-method ddpm --output outputs/generated
```

Для генерации обучать модель заново не нужно: готовые веса приложены. Команды загрузки данных и обучения находятся в [инструкции HW2](hw2/README.md).

## Быстрый запуск HW3

Нужны Linux/WSL2, Docker, NVIDIA GPU и NVIDIA Container Toolkit. Сборка и запуск выбранной конфигурации выполняются одной командой:

```bash
bash hw3/run.sh optimized
```

Проверка HTTP и gRPC:

```bash
bash hw3/run.sh client examples/dog.jpg examples/cat.jpg
bash hw3/run.sh client examples/dog.jpg examples/cat.jpg --protocol grpc
bash hw3/run.sh stop
```

Инструкция по профилированию и сравнение конфигураций — в [отчёте HW3](hw3/README.md).

## Демо и результаты

- [HW1: запись работы приложения на открытом тестовом видео](hw1/demo.mp4).
- [HW2: генерация четырёх изображений методом DDPM](hw2/results/large/demo.mp4).
- [HW2: сравнение DDPM и DDIM](hw2/results/large/comparison.png).
- [HW3: сводный отчёт Model Analyzer](hw3/results/model_analyzer/reports/summaries/resnet18/result_summary.pdf).

Время в таблице HW2 относится к генерации на RTX 3050 Laptop GPU, а длительность демо — к ускоренному показу процесса. Описание измерений и ограничения качества приведены в отчётах.
