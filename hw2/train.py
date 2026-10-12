"""Обучение DDPM с EMA на few-shot-cat или папке изображений."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from utils import create_model, generate, get_device, image_grid, synchronize

import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class CatDataset(Dataset):
    def __init__(self, size, cache_dir):
        data = load_dataset("huggan/few-shot-cat", split="train", cache_dir=str(cache_dir))
        transform = transforms.Compose([
            transforms.Resize((size, size)), transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        self.images = torch.stack([transform(item["image"].convert("RGB")) for item in data])
        self.fingerprint = data._fingerprint

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.get_image(index)
        return image.flip(-1) if torch.rand(()) < 0.5 else image

    def get_image(self, index):
        return self.images[index]


class FolderDataset(Dataset):
    """Читаем только нужную картинку; весь датасет не занимает RAM."""
    def __init__(self, folder, size, augment=True):
        self.folder = folder.resolve()
        self.paths = sorted(p for p in self.folder.rglob('*')
                            if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'})
        if not self.paths:
            raise ValueError(f'В папке нет изображений: {folder}')
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.Resize((size, size)), transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        # Fingerprint зависит от содержимого, а не от пути/времени распаковки.
        digest = hashlib.sha256()
        for path in self.paths:
            digest.update(path.relative_to(self.folder).as_posix().encode() + b'\0')
            with path.open('rb') as file:
                while chunk := file.read(1024 * 1024):
                    digest.update(chunk)
        self.fingerprint = digest.hexdigest()

    def __len__(self):
        return len(self.paths)

    def get_image(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert('RGB'))

    def __getitem__(self, index):
        image = self.get_image(index)
        return image.flip(-1) if self.augment and torch.rand(()) < 0.5 else image


def save_loss(history, output, plot=False):
    with (output / "loss.csv").open("w", newline="", encoding="utf8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.plot([row["step"] for row in history], [row["loss"] for row in history])
        axis.set(xlabel="Optimizer updates", ylabel="Training noise MSE", title="DDPM training, cats")
        axis.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output / "loss.png", dpi=150)
        plt.close(fig)


@torch.inference_mode()
def noise_metrics(model, scheduler, images, device):
    """Диагностика на фиксированных 8 изображениях и шуме, отдельно train/val."""
    images = images[:8].to(device)
    noise = torch.randn(images.shape, generator=torch.Generator().manual_seed(123)).to(device)
    metrics = {}
    for timestep in (0, 100, 500, 750, 999):
        t = torch.full((len(images),), timestep, dtype=torch.long, device=device)
        noisy = scheduler.add_noise(images, noise, t)
        predicted = model(noisy, t).sample
        alpha = scheduler.alphas_cumprod[timestep].to(device)
        restored = (noisy - (1 - alpha).sqrt() * predicted) / alpha.sqrt()
        metrics[str(timestep)] = {
            "noise_mse": F.mse_loss(predicted, noise).item(),
            "x0_mse_unclipped": F.mse_loss(restored, images).item(),
            "x0_mse_clipped": F.mse_loss(restored.clamp(-1, 1), images).item(),
        }
    return metrics


def train(args):
    deadline = (time.perf_counter() + args.time_limit_minutes * 60
                if args.time_limit_minutes else None)
    stop_reason = 'target_steps'
    device = get_device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "loss.csv").exists() and args.resume is None:
        raise ValueError("В папке уже есть обучение. Укажи --resume или другую --output.")
    dataset = (FolderDataset(args.train_dir, args.image_size) if args.train_dir
               else CatDataset(args.image_size, Path(__file__).resolve().parent / "data"))
    validation = FolderDataset(args.val_dir, args.image_size, augment=False) if args.val_dir else None
    if validation and set(dataset.paths) & set(validation.paths):
        raise ValueError('Train и validation содержат одни и те же файлы.')
    preview_images = torch.stack([dataset.get_image(i) for i in range(min(16, len(dataset)))])
    diagnostic_images = preview_images[:8]
    validation_images = (torch.stack([validation.get_image(i) for i in range(min(8, len(validation)))])
                         if validation else None)
    image_grid(preview_images, columns=4).save(output / "dataset_preview.png")
    if validation:
        image_grid(validation_images, columns=4).save(output / 'validation_preview.png')
    config = {
        "image_size": args.image_size, "model_size": args.model_size,
        "batch_size": args.batch_size, "lr": args.lr,
        "seed": args.seed, "max_steps": args.max_steps, "warmup_steps": args.warmup_steps,
        "ema_decay": args.ema_decay, "save_every": args.save_every,
        "dataset": args.dataset_name or ("imagefolder" if args.train_dir else "huggan/few-shot-cat"),
        "dataset_size": len(dataset),
        "dataset_fingerprint": dataset.fingerprint,
        "train_dir": str(args.train_dir) if args.train_dir else None,
        "val_dir": str(args.val_dir) if args.val_dir else None,
        "validation_size": len(validation) if validation else 0,
        "validation_fingerprint": validation.fingerprint if validation else None,
        "milestones": args.milestones,
        "time_limit_minutes": args.time_limit_minutes,
        "precision": "fp16" if device.type == "cuda" else "fp32",
    }
    state = None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state['training_config'].get('model_size', 'base') != args.model_size:
            raise ValueError('Для resume размер модели должен совпадать с checkpoint.')
        if "optimizer" not in state or "ema" not in state:
            raise ValueError("Для новой архитектуры нужен новый запуск; resume принимает новый latest.pt.")
        # Сохраняем тот же график LR и порядок данных при возобновлении.
        for key in ("image_size", "batch_size", "lr", "seed", "max_steps", "warmup_steps",
                    "ema_decay", "dataset_fingerprint"):
            if key == 'max_steps' and args.retarget_steps is not None:
                continue
            if state["training_config"][key] != config[key]:
                raise ValueError(f"Для resume параметр {key} должен совпадать с config.json.")
        if state['training_config'].get('validation_fingerprint') != config['validation_fingerprint']:
            raise ValueError('Для resume validation должен совпадать с исходным запуском.')
        config['lr_retargets'] = list(state['training_config'].get('lr_retargets', []))
        if args.retarget_steps is not None:
            if args.max_steps <= state['global_step']:
                raise ValueError('Новый бюджет должен быть больше уже сохранённого global_step.')
            config['lr_retargets'].append({
                'at_step': state['global_step'],
                'previous_max_steps': state['training_config']['max_steps'],
                'new_max_steps': args.max_steps,
            })
        model = UNet2DModel.from_config(state["model_config"])
        model.load_state_dict(state["model"])
        noise_scheduler = DDPMScheduler.from_config(state["noise_config"])
    else:
        model = create_model(args.image_size, args.model_size)
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear",
                                       prediction_type="epsilon")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = EMAModel(model.parameters(), decay=args.ema_decay, use_ema_warmup=True, foreach=True)
    ema.to(device)
    history, step, epoch, offset, skipped = [], 0, 1, 0, 0
    if state:
        optimizer.load_state_dict(state["optimizer"])
        if args.retarget_steps is not None:
            # Веса/AdamW/EMA продолжаются, но cosine теперь заканчивается
            # на новом общем шаге. last_epoch учитывает уже выполненные updates.
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps,
                num_training_steps=args.max_steps, last_epoch=state['global_step'] - 1)
        else:
            lr_scheduler.load_state_dict(state["lr_scheduler"])
        scaler.load_state_dict(state["scaler"])
        ema.load_state_dict(state["ema"])
        ema.to(device)
        history, step = state["history"], state["global_step"]
        epoch, offset, skipped = state["next_epoch"], state["next_offset"], state["skipped_batches"]
        torch.set_rng_state(state["rng_state"])
        if device.type == "cuda" and state["cuda_rng_state"]:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    if step >= args.max_steps:
        raise ValueError("Этот запуск уже закончил все max-steps обновления.")
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf8")
    parameters = sum(p.numel() for p in model.parameters())
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Device: {name}; model: {parameters:,} parameters; dataset: {len(dataset)}; "
          f"optimizer updates: {step}/{args.max_steps}", flush=True)
    (output / 'train.pid').write_text(f'{os.getpid()}\n', encoding='utf8')
    (output / 'progress.json').write_text(json.dumps({
        'status': 'running', 'pid': os.getpid(), 'step': step,
        'max_steps': args.max_steps, 'epoch': epoch,
    }, indent=2), encoding='utf8')
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    loss_sum, examples, batches = 0.0, 0, 0
    time_limit_reached = False
    while step < args.max_steps and not time_limit_reached:
        # Перестановка зависит только от эпохи. При resume берём оставшуюся часть,
        # не читаем пропущенные картинки и не тратим случайные числа на их flips.
        order_rng = torch.Generator().manual_seed(args.seed + epoch)
        order = torch.randperm(len(dataset), generator=order_rng).tolist()
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=order[offset:],
                            num_workers=0, pin_memory=device.type == "cuda", generator=order_rng)
        model.train()
        for batch in loader:
            current_epoch = epoch
            images = batch.to(device, non_blocking=True)
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, 1000, (len(images),), device=device, dtype=torch.long)
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                predicted_noise = model(noisy_images, timesteps).sample
                loss = F.mse_loss(predicted_noise.float(), noise.float())
            if not torch.isfinite(loss):
                raise RuntimeError("Loss стал NaN/Inf; обучение остановлено.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            offset += len(images)
            if offset == len(dataset):
                epoch, offset = epoch + 1, 0
            batches += 1
            # GradScaler может пропустить обновление при переполнении FP16.
            if scaler.get_scale() < previous_scale:
                skipped += 1
                continue
            step += 1
            lr_scheduler.step()
            ema.step(model.parameters())
            loss_sum += loss.item() * len(images)
            examples += len(images)
            time_limit_reached = deadline is not None and time.perf_counter() >= deadline
            if time_limit_reached:
                stop_reason = 'time_limit'
            save_now = (step % args.save_every == 0 or step in args.milestones
                        or step == args.max_steps or time_limit_reached)
            if step % args.log_every == 0 or save_now:
                synchronize(device)
                seconds = time.perf_counter() - started
                allocated = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0
                reserved = torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0
                row = {"step": step, "epoch": current_epoch, "loss": loss_sum / examples,
                       "lr": optimizer.param_groups[0]["lr"], "seconds": seconds,
                       "batches": batches, "examples": examples, "skipped_batches": skipped,
                       "peak_allocated_mb": allocated, "peak_reserved_mb": reserved}
                history.append(row)
                save_loss(history, output)
                (output / 'progress.json').write_text(json.dumps({
                    'status': 'running', 'pid': os.getpid(), 'step': step, 'max_steps': args.max_steps,
                    'epoch': current_epoch, 'loss': row['loss'], 'lr': row['lr'],
                    'training_seconds': sum(item['seconds'] for item in history),
                    'skipped_batches': skipped,
                }, indent=2), encoding='utf8')
                print(f"Step {step}/{args.max_steps} | epoch {current_epoch} | MSE {row['loss']:.5f} | "
                      f"lr {row['lr']:.2g} | {seconds:.1f}s | peak {allocated:.0f} MiB", flush=True)
                loss_sum, examples, batches = 0.0, 0, 0
            if save_now:
                checkpoint = {
                    "model": model.state_dict(), "weights": "raw", "model_config": dict(model.config),
                    "noise_config": dict(noise_scheduler.config), "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "ema": ema.state_dict(), "global_step": step, "epoch": current_epoch,
                    "next_epoch": epoch, "next_offset": offset, "skipped_batches": skipped,
                    "history": history, "training_config": config, "rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                }
                torch.save(checkpoint, output / "latest.tmp")
                (output / "latest.tmp").replace(output / "latest.pt")
                # Временно подставляем EMA для компактной модели и визуальной проверки.
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                try:
                    torch.save({"model": model.state_dict(), "weights": "ema",
                                "model_config": dict(model.config), "noise_config": dict(noise_scheduler.config),
                                "global_step": step, "epoch": current_epoch, "training_config": config},
                               output / "model.tmp")
                    (output / "model.tmp").replace(output / "model.pt")
                    if step in args.milestones or step == args.max_steps or time_limit_reached:
                        milestones = output / 'milestones'
                        milestones.mkdir(exist_ok=True)
                        shutil.copyfile(output / 'model.pt', milestones / f'step_{step:06d}.pt')
                    generated, _, _ = generate(model, noise_scheduler.config, steps=50, count=4,
                                                seed=args.seed, device=device)
                    previews = output / "previews"
                    previews.mkdir(exist_ok=True)
                    image_grid(generated, columns=2).save(previews / f"step_{step:06d}.png")
                    metrics = {'step': step, 'weights': 'ema',
                               'timesteps': noise_metrics(model, noise_scheduler, diagnostic_images, device)}
                    if validation:
                        metrics['validation_timesteps'] = noise_metrics(
                            model, noise_scheduler, validation_images, device)
                        metrics['validation_images'] = len(validation_images)
                    (previews / f"step_{step:06d}_metrics.json").write_text(
                        json.dumps(metrics, indent=2),
                        encoding="utf8")
                finally:
                    ema.restore(model.parameters())
                    model.train()
                save_loss(history, output, plot=True)
                print(f"Saved EMA preview and checkpoint at step {step}", flush=True)
            if step % args.log_every == 0 or save_now:
                # Не включаем сохранение, генерацию и диагностику в время обучения.
                synchronize(device)
                started = time.perf_counter()
            if step == args.max_steps or time_limit_reached:
                break
    summary = {
        **config, "device": name, "parameters": parameters, "optimizer_steps": step,
        "skipped_batches": skipped, "last_epoch": current_epoch,
        "training_seconds": sum(row["seconds"] for row in history), "final_loss": history[-1]["loss"],
        "peak_allocated_mb": max(row["peak_allocated_mb"] for row in history),
        "peak_reserved_mb": max(row["peak_reserved_mb"] for row in history),
        "stop_reason": stop_reason,
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    (output / 'progress.json').write_text(json.dumps({
        'status': 'time_limit' if time_limit_reached else 'complete',
        'stop_reason': stop_reason, 'pid': os.getpid(), 'step': step, 'max_steps': args.max_steps,
        'epoch': current_epoch, 'loss': history[-1]['loss'],
        'training_seconds': summary['training_seconds'], 'skipped_batches': skipped,
    }, indent=2), encoding='utf8')
    print(f"Saved: {output / 'model.pt'}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=20000, help="Общее число обновлений optimizer")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument('--model-size', choices=('base', 'large'), default='base',
                        help='base: 16.89M параметров; large: 33.89M, примерно вдвое больше')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument('--train-dir', type=Path, help='Папка обучающих изображений; без неё few-shot-cat')
    parser.add_argument('--val-dir', type=Path, help='Отдельная папка validation, без аугментации')
    parser.add_argument('--dataset-name', help='Название данных для config.json')
    parser.add_argument('--milestones', nargs='+', type=int, default=[20000, 50000, 100000])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument('--retarget-steps', type=int,
                        help='С resume: изменить общий бюджет и пересчитать cosine LR с текущего шага')
    parser.add_argument('--time-limit-minutes', type=float,
                        help='Лимит времени этого запуска; сохранить последний шаг перед остановкой')
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.retarget_steps is not None:
        if not args.resume:
            parser.error('--retarget-steps требует --resume')
        args.max_steps = args.retarget_steps
    if args.time_limit_minutes is not None and args.time_limit_minutes <= 0:
        parser.error('time-limit-minutes должен быть положительным')
    if args.output is None:
        args.output = Path('runs/afhqv2_cats128' if args.train_dir else 'runs/cats128_improved')
        if args.model_size == 'large':
            args.output = args.output.with_name(args.output.name + '_large')
    if args.val_dir and not args.train_dir:
        parser.error('--val-dir используется вместе с --train-dir')
    if any(step <= 0 for step in args.milestones):
        parser.error('milestones должны быть положительными')
    if min(args.max_steps, args.batch_size, args.save_every, args.log_every) < 1 or args.lr <= 0:
        parser.error("max-steps, batch-size, save-every, log-every и lr должны быть положительными")
    if args.image_size < 32 or args.image_size % 16:
        parser.error("image-size должен быть не меньше 32 и кратен 16")
    if not 0 <= args.warmup_steps < args.max_steps:
        parser.error("warmup-steps должен быть неотрицательным и меньше max-steps")
    if not 0 < args.ema_decay < 1:
        parser.error("ema-decay должен лежать между 0 и 1")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    try:
        train(arguments)
    except BaseException as error:
        progress_path = arguments.output / 'progress.json'
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding='utf8'))
            # Ошибочный повторный запуск не меняет статус чужого активного процесса.
            if progress.get('pid') == os.getpid():
                progress['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
                progress['error'] = f'{type(error).__name__}: {error}'
                progress_path.write_text(json.dumps(progress, indent=2), encoding='utf8')
        raise
