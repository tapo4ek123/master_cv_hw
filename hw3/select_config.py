"""Скопировать выбранную конфигурацию Model Analyzer для обычного запуска."""
import argparse
import re
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('config', type=Path, help='Путь к config.pbtxt выбранного варианта')
args = parser.parse_args()
text = args.config.read_text(encoding='utf8')
# Analyzer переименовывает варианты; клиент обращается к исходному resnet18.
text, count = re.subn(r'^name:\s*"[^"]+"', 'name: "resnet18"', text, count=1, flags=re.MULTILINE)
if count != 1:
    raise ValueError('В config.pbtxt не найдено имя модели')
target = Path(__file__).parent / 'configs/optimized.pbtxt'
target.write_text(text, encoding='utf8')
print(f'Выбрана конфигурация: {args.config} -> {target}')
