"""Удаление фона с камеры или видео с помощью MediaPipe (CPU)."""

import argparse
import json
import os
import platform
import shutil
import sys
import time
import urllib.request
from pathlib import Path

# В WSL драйвер Mesa/D3D12 может падать при закрытии контекста MediaPipe.
# Сегментация работает на CPU, поэтому используем программный контекст.
if "microsoft" in platform.release().lower():
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import cv2
import mediapipe as mp
import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / "models" / "selfie_segmenter_landscape.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite"
)


def parse_color(value):
    """В параметрах цвет задаётся как R,G,B; OpenCV использует B,G,R."""
    try:
        rgb = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Цвет должен иметь вид 40,120,180") from error
    if len(rgb) != 3 or not all(0 <= channel <= 255 for channel in rgb):
        raise argparse.ArgumentTypeError("Нужны три значения цвета от 0 до 255")
    return rgb[::-1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Номер камеры или путь к видео")
    parser.add_argument("--mode", choices=("color", "blur", "image"), default="color")
    parser.add_argument("--color", type=parse_color, default=parse_color("60,120,180"))
    parser.add_argument("--background", type=Path, help="Изображение для режима image")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--smooth", type=float, default=0.8,
                        help="Вес текущей маски: 1 отключает сглаживание во времени")
    parser.add_argument("--output", type=Path, help="Сохранить результат в MP4")
    parser.add_argument("--compare", action="store_true", help="Показывать исходник рядом")
    parser.add_argument("--no-display", action="store_true", help="Обработка без окна")
    parser.add_argument('--realtime', action='store_true',
                        help='Показывать видеофайл с частотой его кадров, без ускорения')
    parser.add_argument("--max-frames", type=int, default=0, help="0 — обработать всё видео")
    parser.add_argument("--stats", type=Path, default=Path("metrics.json"))
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        parser.error("Ширина и высота должны быть положительными")
    if not 0 < args.smooth <= 1:
        parser.error("--smooth должен быть больше 0 и не больше 1")
    if args.max_frames < 0:
        parser.error("--max-frames не может быть отрицательным")
    if args.mode == "image" and args.background is None:
        parser.error("Для режима image нужен --background")
    if args.output is not None and args.output.suffix.lower() != ".mp4":
        parser.error("В --output укажи файл с расширением .mp4")
    if not args.source.isdecimal() and args.output is not None:
        if Path(args.source).resolve() == args.output.resolve():
            parser.error("Входное и выходное видео должны быть разными файлами")
    return args


def prepare_model():
    if MODEL_PATH.is_file():
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_PATH.with_suffix(".part")
    print("Скачивание модели MediaPipe...")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        temporary.replace(MODEL_PATH)
    finally:
        temporary.unlink(missing_ok=True)


def get_cpu_name():
    if sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor() or "unknown"


def composite(frame, mask, mode, background):
    if mode == "blur":
        # Размываем уменьшенный кадр: это дешевле большого ядра на полном кадре.
        height, width = frame.shape[:2]
        small = cv2.resize(frame, (max(1, width // 4), max(1, height // 4)))
        small = cv2.GaussianBlur(small, (21, 21), 0)
        background = cv2.resize(small, (width, height))
    alpha = mask[:, :, np.newaxis]
    return (frame * alpha + background * (1 - alpha)).astype(np.uint8)


def run(args):
    prepare_model()
    size = (args.width, args.height)
    background = np.full((args.height, args.width, 3), args.color, dtype=np.uint8)
    if args.mode == "image":
        image = cv2.imread(str(args.background))
        if image is None:
            raise ValueError(f"Не удалось прочитать фон: {args.background}")
        background = cv2.resize(image, size)

    is_camera = args.source.isdecimal()
    source = int(args.source) if is_camera else args.source
    capture = cv2.VideoCapture(source)
    writer = None
    frames = 0
    times = []
    previous_mask = None
    last_timestamp = -1
    warmup = 5
    measured_start = None
    playback_start = None
    input_size = None
    started = time.perf_counter()
    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH), delegate=mp.tasks.BaseOptions.Delegate.CPU
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_category_mask=False,
        output_confidence_masks=True,
    )
    try:
        if not capture.isOpened():
            raise ValueError(f"Не удалось открыть источник: {args.source}")
        if is_camera:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(source_fps) or source_fps <= 0:
            source_fps = 30.0

        with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
            while not args.max_frames or frames < args.max_frames:
                if frames == warmup:
                    measured_start = time.perf_counter()
                success, frame = capture.read()
                if not success:
                    if frames == 0 or is_camera:
                        raise ValueError("Не удалось получить кадр из источника")
                    break
                if input_size is None:
                    input_size = [frame.shape[1], frame.shape[0]]

                frame_start = time.perf_counter()
                if playback_start is None:
                    playback_start = frame_start
                frame = cv2.resize(frame, size)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = (int((frame_start - started) * 1000) if is_camera
                             else int(frames * 1000 / source_fps))
                timestamp = max(timestamp, last_timestamp + 1)
                last_timestamp = timestamp
                result = segmenter.segment_for_video(image, timestamp)
                # Эта версия SelfieSegmenter возвращает одну карту вероятности человека.
                mask = result.confidence_masks[0].numpy_view().copy()
                mask = cv2.resize(mask, size)
                mask = np.clip((mask - 0.2) / 0.6, 0, 1)
                mask = cv2.GaussianBlur(mask, (3, 3), 0)
                if previous_mask is not None:
                    mask = args.smooth * mask + (1 - args.smooth) * previous_mask
                previous_mask = mask
                output = composite(frame, mask, args.mode, background)
                elapsed = time.perf_counter() - frame_start
                if frames >= warmup:
                    times.append(elapsed)
                frames += 1

                # В замер обработки не входят надпись, запись файла и окно.
                current_fps = 1 / np.mean(times[-30:]) if times else 1 / elapsed
                if args.compare:
                    output = np.hstack((frame, output))
                cv2.putText(output, f"CPU | processing: {current_fps:.1f} FPS",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                if args.output is not None:
                    if writer is None:
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 source_fps, (output.shape[1], output.shape[0]))
                        if not writer.isOpened():
                            raise ValueError(f"Не удалось создать видео: {args.output}")
                    writer.write(output)
                if not args.no_display:
                    cv2.imshow("Background removal (Q / Esc to exit)", output)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                if args.realtime and not is_camera:
                    # Ожидание относится к воспроизведению, а не к обработке кадра.
                    delay = playback_start + frames / source_fps - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
        finished = time.perf_counter()
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    total_processing = sum(times)
    stats = {
        "source": args.source,
        "mode": args.mode,
        "cpu": get_cpu_name(),
        "platform": platform.platform(),
        "versions": {"python": platform.python_version(), "mediapipe": mp.__version__,
                     "opencv": cv2.__version__, "numpy": np.__version__},
        "input_resolution": input_size,
        "processing_resolution": list(size),
        "source_fps": source_fps,
        "frames_processed": frames,
        "warmup_frames": min(frames, warmup),
        "frames_measured": len(times),
        "processing_fps": len(times) / total_processing if times else None,
        "mean_latency_ms": np.mean(times) * 1000 if times else None,
        "p95_latency_ms": float(np.percentile(times, 95)) * 1000 if times else None,
        "throughput_fps": len(times) / (finished - measured_start) if times else None,
        "measurement": "Processing includes resize, RGB conversion, segmentation, mask smoothing "
                       "and compositing. Throughput also includes capture, display and recording. "
                       "First 5 frames are excluded as warmup.",
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"Обработано кадров: {frames}; разрешение обработки: {args.width}x{args.height}")
    if times:
        print(f"Обработка: {stats['processing_fps']:.1f} FPS; "
              f"среднее время: {stats['mean_latency_ms']:.1f} мс")
        print(f"Весь цикл (с захватом/показом/записью): {stats['throughput_fps']:.1f} FPS")
    else:
        print("Для замера скорости нужно больше 5 кадров.")
    print(f"Замеры сохранены: {args.stats}")


if __name__ == "__main__":
    arguments = parse_args()
    try:
        run(arguments)
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Обработка остановлена.")
