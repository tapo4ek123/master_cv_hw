"""Скачать AFHQ v2 cat; официальный test/cat сохранить отдельно как val/cat."""
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image


URL = 'https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=1'
SOURCE = 'https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq'


def download(path):
    offset = path.stat().st_size if path.exists() else 0
    request = urllib.request.Request(URL, headers={'Range': f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(request, timeout=60) as response:
        resumed = offset and response.status == 206
        if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
            raise ValueError('Сервер вернул неправильную часть архива.')
        done = offset if resumed else 0
        length = int(response.headers.get('Content-Length', 0))
        total = done + length
        with path.open('ab' if resumed else 'wb') as file:
            while chunk := response.read(4 * 1024 * 1024):
                file.write(chunk)
                done += len(chunk)
                print(f'Downloaded {done / 2**20:.0f} / {total / 2**20:.0f} MiB', flush=True)


def prepare(output, archive):
    counts = {'train': 0, 'val': 0}
    names = {'train': set(), 'val': set()}
    split_map = {'train': 'train', 'test': 'val', 'val': 'val'}
    original_splits = {'train': set(), 'val': set()}
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            path = PurePosixPath(item.filename)
            parts = path.parts
            if (len(parts) < 3 or parts[-3] not in split_map or parts[-2] != 'cat'
                    or path.suffix.lower() != '.png' or '__MACOSX' in parts
                    or path.name.startswith('.')):
                continue
            split = split_map[parts[-3]]
            original_splits[split].add(parts[-3])
            if path.name in names[split]:
                raise ValueError(f'Повторное имя в архиве: {item.filename}')
            names[split].add(path.name)
            target = output / split / 'cat' / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.part')
            # Чтение до конца также проверяет CRC ZIP. Пути из архива не извлекаются напрямую.
            with source.open(item) as image_file, temporary.open('wb') as file:
                shutil.copyfileobj(image_file, file)
            with Image.open(temporary) as image:
                if image.size != (512, 512):
                    raise ValueError(f'Неожиданное разрешение AFHQ v2: {item.filename}, {image.size}')
                image.convert('RGB').load()
            temporary.replace(target)
            counts[split] += 1
            if sum(counts.values()) % 250 == 0:
                print(f'Extracted cats: {counts}', flush=True)
    if not all(counts.values()):
        raise ValueError('В архиве не найдены оба разбиения train/cat и val/cat.')
    if names['train'] & names['val']:
        raise ValueError('Имена изображений пересекаются между train и validation.')
    digest = hashlib.sha256()
    with archive.open('rb') as file:
        while chunk := file.read(4 * 1024 * 1024):
            digest.update(chunk)
    metadata = {'dataset': 'AFHQ v2 cat', 'source': SOURCE, 'download_url': URL,
                'license': 'CC BY-NC 4.0, NAVER Corporation',
                'archive_sha256': digest.hexdigest(), 'counts': counts,
                'original_resolution': [512, 512],
                'original_splits': {key: sorted(value) for key, value in original_splits.items()},
                'validation': 'Official held-out cat split, saved as val/cat; excluded from training'}
    (output / 'dataset_info.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/afhq_v2'))
    parser.add_argument('--archive', type=Path, help='Готовый ZIP; иначе скачать с официального источника')
    parser.add_argument('--keep-archive', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.archive or args.output / 'archive.zip.part'
    if args.archive is None:
        download(archive)
    prepare(args.output, archive)
    # Внешний пользовательский архив не удаляем. Удаляется только наша загрузка.
    if args.archive is None and not args.keep_archive:
        archive.unlink()


if __name__ == '__main__':
    main()
