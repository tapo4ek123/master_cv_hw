"""Восстановить готовые веса из приложенных частей, без скачивания и обучения."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    folder = Path(__file__).resolve().parent / 'models'
    manifest = json.loads((folder / 'manifest.json').read_text())
    for model in manifest['models']:
        target = folder / model['target']
        if target.is_file() and sha256(target) == model['sha256']:
            print(f"Готово: models/{model['target']}")
            continue
        temporary = target.with_suffix('.tmp')
        digest = hashlib.sha256()
        try:
            with temporary.open('wb') as output:
                for entry in model['parts']:
                    part = folder / entry['file']
                    if part.stat().st_size != entry['size_bytes'] or sha256(part) != entry['sha256']:
                        raise ValueError(f'Повреждена часть весов: {part.name}')
                    with part.open('rb') as file:
                        while block := file.read(1024 * 1024):
                            digest.update(block)
                            output.write(block)
            if temporary.stat().st_size != model['size_bytes'] or digest.hexdigest() != model['sha256']:
                raise ValueError(f"Не совпала контрольная сумма: {model['name']}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Восстановлено: models/{model['target']}")


if __name__ == '__main__':
    main()
