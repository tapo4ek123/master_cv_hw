"""Общие функции для обучения и генерации изображений котов."""

import math
import os
import time
from pathlib import Path

# Используем только PyTorch, даже если в окружении установлен TensorFlow.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from PIL import Image, ImageDraw


def get_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA недоступна. Установи PyTorch с CUDA или выбери --device cpu.")
    return torch.device(name)


MODEL_CHANNELS = {
    "base": (32, 64, 128, 128, 256),
    "large": (48, 88, 184, 184, 360),
}


def create_model(image_size=128, model_size="base"):
    return UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        # Четыре уменьшения разрешения: 128 -> 64 -> 32 -> 16 -> 8.
        block_out_channels=MODEL_CHANNELS[model_size],
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D",
                          "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
                        "UpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
    )


def load_model(checkpoint, device, weights="auto"):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = UNet2DModel.from_config(state["model_config"])
    model.load_state_dict(state["model"])
    # latest.pt содержит обычные веса и EMA; model.pt — уже готовые EMA-веса.
    saved_weights = state.get("weights", "raw")
    selected = ("ema" if "ema" in state else saved_weights) if weights == "auto" else weights
    if selected == "ema" and "ema" in state:
        from diffusers.training_utils import EMAModel
        ema = EMAModel(model.parameters())
        ema.load_state_dict(state["ema"])
        ema.copy_to(model.parameters())
    elif selected != saved_weights:
        raise ValueError(f"В checkpoint нет весов {selected}; используй latest.pt или --weights auto.")
    state["selected_weights"] = selected
    return model.to(device).eval(), state


def image_grid(images, columns=4, label=None, scale=1):
    """Преобразует тензоры в [-1, 1] в сетку PIL без matplotlib."""
    pixels = ((images.detach().float().cpu().clamp(-1, 1) + 1) / 2 * 255)
    pixels = pixels.round().byte().permute(0, 2, 3, 1).numpy()
    height, width = pixels.shape[1:3]
    columns = min(columns, len(pixels))
    rows = math.ceil(len(pixels) / columns)
    header = 24 if label else 0
    grid = Image.new("RGB", (columns * width, rows * height + header), "white")
    for i, array in enumerate(pixels):
        grid.paste(Image.fromarray(array), ((i % columns) * width, (i // columns) * height + header))
    if label:
        ImageDraw.Draw(grid).text((6, 6), label, fill="black")
    if scale != 1:
        grid = grid.resize((grid.width * scale, grid.height * scale), Image.Resampling.NEAREST)
    return grid


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate(model, scheduler_config, method="ddim", steps=50, count=8,
             seed=42, device=None, progress=False, collect_frames=False):
    """Один и тот же начальный шум для любого метода при одинаковом seed."""
    device = device or next(model.parameters()).device
    scheduler_class = DDPMScheduler if method == "ddpm" else DDIMScheduler
    scheduler = scheduler_class.from_config(scheduler_config)
    scheduler.set_timesteps(steps, device=device)
    size = model.config.sample_size
    # Отдельные генераторы не меняют случайные числа основного цикла обучения.
    initial_generator = torch.Generator(device="cpu").manual_seed(seed)
    images = torch.randn((count, 3, size, size), generator=initial_generator).to(device)
    step_generator = torch.Generator(device=device).manual_seed(seed + 1)
    frames = [images.cpu()] if collect_frames else []
    iterator = scheduler.timesteps
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"{method.upper()} ({steps} steps)")
    model.eval()
    synchronize(device)
    started = time.perf_counter()
    for i, timestep in enumerate(iterator):
        # Обратный процесс считаем в FP32: ошибки на шумных шагах усиливаются.
        predicted_noise = model(images, timestep).sample
        # DDPM добавляет случайный шум на обратных шагах, DDIM с eta=0 — нет.
        if method == "ddpm":
            images = scheduler.step(predicted_noise, timestep, images,
                                    generator=step_generator).prev_sample
        else:
            # После clipping x0 пересчитываем шум, чтобы части обновления согласовывались.
            images = scheduler.step(predicted_noise, timestep, images, eta=0,
                                    use_clipped_model_output=True).prev_sample
        if collect_frames and ((i + 1) % max(1, steps // 60) == 0 or i == steps - 1):
            frames.append(images.cpu())
    synchronize(device)
    seconds = time.perf_counter() - started
    return images.cpu(), seconds, frames


def save_demo(frames, path, method, steps):
    """12-секундное MP4: движение от шума к итоговой сетке."""
    import imageio.v2 as imageio
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=10, codec="libx264", macro_block_size=1) as writer:
        for i in range(120):
            # Последние две секунды показываем конечный результат.
            index = min(len(frames) - 1, round(i / 99 * (len(frames) - 1)))
            label = f"{method.upper()}, {steps} steps | noise -> image"
            grid = image_grid(frames[index], columns=4, label=label, scale=2)
            writer.append_data(np.asarray(grid))
