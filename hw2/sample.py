"""Генерация котов и сравнение DDPM / DDIM на одной обученной модели."""

import argparse
import csv
import json
from pathlib import Path

from utils import generate, get_device, image_grid, load_model, save_demo

import torch
from PIL import Image


def run(args):
    device = get_device(args.device)
    torch.set_num_threads(4)
    model, state = load_model(args.checkpoint, device, args.weights)
    maximum = state["noise_config"]["num_train_timesteps"]
    if args.steps > maximum:
        raise ValueError(f"Число шагов не должно быть больше {maximum}.")
    args.output.mkdir(parents=True, exist_ok=True)
    settings = [("ddpm", 1000), ("ddim", 200), ("ddim", 50)] if args.compare else [(args.method, args.steps)]
    results, grids = [], []
    for method, steps in settings:
        # Прогрев CUDA перед замером; тот же batch size и размер изображения.
        generate(model, state["noise_config"], method, steps=2, count=args.count,
                 seed=args.seed, device=device)
        durations = []
        images = None
        for _ in range(args.repeats):
            images, seconds, _ = generate(model, state["noise_config"], method, steps,
                                         args.count, args.seed, device, progress=True)
            durations.append(seconds)
        grid = image_grid(images, columns=4, label=f"{method.upper()} | {steps} steps")
        grid.save(args.output / f"{method}_{steps}.png")
        grids.append(grid)
        for i, image in enumerate(images):
            image_grid(image.unsqueeze(0), columns=1).save(args.output / f"{method}_{steps}_{i:02d}.png")
        mean = sum(durations) / len(durations)
        results.append({"method": method, "steps": steps, "images": args.count,
                        "mean_seconds": mean, "seconds_per_image": mean / args.count,
                        "repeats": args.repeats, "times_seconds": durations})
        print(f"{method.upper()}, {steps} steps: {mean:.2f}s / {args.count} images", flush=True)
    if args.compare:
        combined = Image.new("RGB", (sum(grid.width for grid in grids), max(grid.height for grid in grids)), "white")
        offset = 0
        for grid in grids:
            combined.paste(grid, (offset, 0))
            offset += grid.width
        combined.save(args.output / "comparison.png")
    summary = {
        "checkpoint": str(args.checkpoint), "epoch": state["epoch"], "seed": args.seed,
        "optimizer_steps": state.get("global_step"), "weights": state["selected_weights"],
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "resolution": model.config.sample_size, "ddim_eta": 0,
        "sampling_precision": "fp32", "ddim_use_clipped_model_output": True,
        "measurement": "Same checkpoint, image count and initial CPU-generated noise. "
                       "CUDA warmup before each method; synchronized generation timing, "
                       "excluding model loading, saving images and demo recording.",
        "results": results,
    }
    (args.output / "timings.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    with (args.output / "timings.csv").open("w", newline="", encoding="utf8") as file:
        fields = [key for key in results[0] if key != "times_seconds"]
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    if args.demo:
        # Видео сохраняется отдельным запуском и не входит в замеры скорости.
        if args.compare:
            method, steps = ("ddpm", 1000) if args.demo_method == "ddpm" else ("ddim", 200)
        else:
            method, steps = settings[0]
        _, _, frames = generate(model, state["noise_config"], method, steps,
                                min(args.count, 4), args.seed, device, collect_frames=True)
        save_demo(frames, args.output / "demo.mp4", method, steps)
        print(f"Saved demo: {args.output / 'demo.mp4'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/afhq_large/model.pt"))
    parser.add_argument("--weights", choices=("auto", "ema", "raw"), default="auto",
                        help="По умолчанию EMA, если они есть; raw доступны в latest.pt")
    parser.add_argument("--method", choices=("ddpm", "ddim"), default="ddim")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/generated"))
    parser.add_argument("--compare", action="store_true", help="DDPM 1000 / DDIM 200 / DDIM 50")
    parser.add_argument("--demo", action="store_true", help="Сохранить MP4 с обратным процессом")
    parser.add_argument('--demo-method', choices=('ddpm', 'ddim'), default='ddim',
                        help='Метод для видео при --compare --demo: DDPM 1000 или DDIM 200')
    args = parser.parse_args()
    if min(args.steps, args.count, args.repeats) < 1:
        parser.error("steps, count и repeats должны быть положительными")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint не найден: {args.checkpoint}")
    return args


if __name__ == "__main__":
    run(parse_args())
